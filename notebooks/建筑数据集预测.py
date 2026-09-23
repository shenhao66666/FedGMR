import sys
import os

from pathlib import Path


parent = Path(os.path.abspath("")).resolve().parents[0]
if parent not in sys.path:
    sys.path.insert(0, str(parent))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import copy
from logging import INFO
from ml.utils.logger import log
import random

from collections import OrderedDict

import numpy as np
import torch

import pandas as pd

from matplotlib import pyplot as plt

from argparse import Namespace
from ml.utils.data_utils import read_data, generate_time_lags, time_to_feature, handle_nans, to_Xy, \
    to_torch_dataset, to_timeseries_rep, assign_statistics, \
    to_train_val, scale_features, get_data_by_area, remove_identifiers, get_exogenous_data_by_area, handle_outliers, split_exogenous_by_indices
from ml.utils.train_utils import train, test

from ml.models.mlp import MLP
from ml.models.rnn import RNN
from ml.models.lstm import LSTM
from ml.models.gru import GRU
from ml.models.cnn import CNN
from ml.models.rnn_autoencoder import DualAttentionAutoEncoder

from ml.fl.defaults import create_regression_client
from ml.fl.client_proxy import SimpleClientProxy
from ml.fl.server.server import Server
from ml.utils.helpers import accumulate_metric
from ml.utils.model_utils import *  # 从 model_utils 导入



print(f"Script arguments: {args}\n")


device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
print(f"Using {device}")


# Outlier detection specification
# 不再写死每个 building 的 kwargs，默认只指定需要检测的列，kwargs 在 preprocess 时根据数据动态生成
if bool(getattr(args, "outlier_detection", False)):
    args.outlier_columns = ['meter_reading']
    args.outlier_kwargs = None


def seed_all():
    # ensure reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_all()


def update_scaler_keys(y_scalers, client_keys):
    """支持 client_keys 为 'area' 或 'area_Client_x' 两种形式，返回以 client_key 为键的 scaler dict。"""
    updated_scalers = {}
    for client_key in client_keys:
        area = client_key.split("_Client_")[0] if "_Client_" in client_key else client_key
        if area in y_scalers:
            updated_scalers[client_key] = y_scalers[area]
        elif client_key in y_scalers:
            updated_scalers[client_key] = y_scalers[client_key]
    return updated_scalers


def make_preprocessing():
    """Preprocess a given .csv"""

    exogenous_data_train = None  # 先初始化，避免UnboundLocalError
    exogenous_data_val = None    # 先初始化，避免UnboundLocalError

    # read data
    df = read_data(args.data_path)
    # handle nans
    df = handle_nans(train_data=df, constant=args.nan_constant,
                     identifier=args.identifier)
    # split to train/validation
    train_data, val_data = to_train_val(df)

    # handle outliers only when explicitly enabled
    if bool(getattr(args, "outlier_detection", False)):
        # 若未预设 kwargs，则根据训练集中出现的所有建筑动态生成默认 (10,90)
        if args.outlier_kwargs is None:
            unique_areas = train_data[args.identifier].unique()
            args.outlier_kwargs = {area: (10, 90) for area in unique_areas}
        train_data = handle_outliers(df=train_data, columns=args.outlier_columns,
                                     identifier=args.identifier, kwargs=args.outlier_kwargs)

    # get X and y
    X_train, X_val, y_train, y_val = to_Xy(train_data=train_data, val_data=val_data,
                                           targets=args.targets)

    # 将时间特征并入 X（在缩放与lag之前，保证参与分组与时序展开）
    if bool(getattr(args, "use_time_features", True)):
        date_time_df_train = time_to_feature(
            X_train, args.use_time_features, identifier=args.identifier
        )
        date_time_df_val = time_to_feature(
            X_val, args.use_time_features, identifier=args.identifier
        )

        if date_time_df_train is not None:
            if args.identifier in date_time_df_train.columns:
                date_time_df_train = date_time_df_train.drop(columns=[args.identifier])
            X_train = pd.concat([X_train, date_time_df_train], axis=1)
            X_train = X_train.loc[:, ~X_train.columns.duplicated()].copy()

        if date_time_df_val is not None:
            if args.identifier in date_time_df_val.columns:
                date_time_df_val = date_time_df_val.drop(columns=[args.identifier])
            X_val = pd.concat([X_val, date_time_df_val], axis=1)
            X_val = X_val.loc[:, ~X_val.columns.duplicated()].copy()

    # 保存缩放前的客户端统计量，供后续重叠共享特征自动选择使用。
    # 静态特征在 per-area scaling 后可能全部变为 0，缩放后的 between 无法再区分建筑上下文强弱。
    def _compute_unscaled_group_feature_stats(df, identifier):
        stats = {}
        if df is None or identifier not in df.columns:
            return stats
        for col in df.columns:
            if col == identifier:
                continue
            tmp = df[[identifier, col]].dropna()
            if tmp.empty or tmp[identifier].nunique() < 2:
                continue
            per_area_stats = tmp.groupby(identifier)[col].agg(["mean", "std"])
            stats[col] = {
                "means": per_area_stats["mean"].astype(float).tolist(),
                "stds": per_area_stats["std"].fillna(0.0).astype(float).tolist(),
                "within": float(per_area_stats["std"].fillna(0.0).mean()),
                "between": float(per_area_stats["mean"].std(ddof=0)),
            }
            stats[col]["ratio"] = stats[col]["within"] / (stats[col]["between"] + 1e-8)
        return stats

    args.group_unscaled_feature_stats = _compute_unscaled_group_feature_stats(
        X_train, args.identifier
    )

    # scale X
    X_train, X_val, x_scalers = scale_features(train_data=X_train, val_data=X_val,
                                               scaler=args.x_scaler,
                                               per_area=True,  # the features are scaled locally
                                               identifier=args.identifier)
    # scale y
    y_train, y_val, y_scalers = scale_features(train_data=y_train, val_data=y_val,
                                               scaler=args.y_scaler,
                                               per_area=True,
                                               identifier=args.identifier)

    # generate time lags
    X_train = generate_time_lags(X_train, args.num_lags)
    X_val = generate_time_lags(X_val, args.num_lags)
    y_train = generate_time_lags(y_train, args.num_lags, is_y=True)
    y_val = generate_time_lags(y_val, args.num_lags, is_y=True)

    # 新增：滑窗后重置索引，保证所有DataFrame索引一致
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    if exogenous_data_train is not None:
        exogenous_data_train = exogenous_data_train.reset_index(drop=True)
    if exogenous_data_val is not None:
        exogenous_data_val = exogenous_data_val.reset_index(drop=True)

    print("原始meter_reading样例：", df["meter_reading"].values[:5])
    print("归一化后y_train样例：", y_train[:5])

    # get statistics as exogenous data
    stats_df_train = assign_statistics(X_train, args.assign_stats, args.num_lags,
                                       targets=args.targets, identifier=args.identifier)
    stats_df_val = assign_statistics(X_val, args.assign_stats, args.num_lags,
                                     targets=args.targets, identifier=args.identifier)

    # concat the exogenous features (if any) to a single dataframe
    if stats_df_train is not None:
        exogenous_data_train = stats_df_train.copy()
        assert len(exogenous_data_train) == len(X_train) == len(y_train)
    else:
        exogenous_data_train = None
    if stats_df_val is not None:
        exogenous_data_val = stats_df_val.copy()
        assert len(exogenous_data_val) == len(X_val) == len(y_val)
    else:
        exogenous_data_val = None

    return X_train, X_val, y_train, y_val, exogenous_data_train, exogenous_data_val, x_scalers, y_scalers


X_train, X_val, y_train, y_val, exogenous_data_train, exogenous_data_val, x_scalers, y_scalers = make_preprocessing()


# 保留一份原始的
original_y_scalers = y_scalers.copy()

print("Original y_scalers keys:", original_y_scalers.keys())


def make_postprocessing(X_train, X_val, y_train, y_val, exogenous_data_train, exogenous_data_val, x_scalers, y_scalers):
    """Make data ready to be fed into ml algorithms"""

    if X_train[args.identifier].nunique() != 1:
        # 使用工具函数按 area 生成数据（clients_per_area=1 表示不拆分子客户端）
        area_X_train, area_X_val, area_y_train, area_y_val, area_train_indices, area_val_indices = get_data_by_area(
            X_train, X_val, y_train, y_val, identifier=args.identifier, clients_per_area=1
        )

        # 外生数据按 area 获取（clients_per_area=1，不会拆分为 "_Client_"）
        if exogenous_data_train is not None:
            area_exogenous_data_train, area_exogenous_data_val = get_exogenous_data_by_area(
                exogenous_data_train, exogenous_data_val, identifier=args.identifier, clients_per_area=1
            )
        else:
            area_exogenous_data_train, area_exogenous_data_val = None, None

        # 将每个 area 的 DataFrame 转为 numpy，并处理可能出现的空分片（reshape 空数组）
        total_cols_full = len(X_train.columns) - (1 if args.identifier in X_train.columns else 0)
        num_features_tmp = total_cols_full // args.num_lags
        total_cols = num_features_tmp * args.num_lags
        for area in list(area_X_train.keys()):
            tmp_X_train, tmp_y_train, tmp_X_val, tmp_y_val = remove_identifiers(
                area_X_train[area], area_y_train[area], area_X_val[area], area_y_val[area]
            )
            tmp_X_train = tmp_X_train.to_numpy()
            tmp_y_train = tmp_y_train.to_numpy()
            tmp_X_val = tmp_X_val.to_numpy()
            tmp_y_val = tmp_y_val.to_numpy()

            if tmp_X_train.size == 0:
                tmp_X_train = tmp_X_train.reshape((0, total_cols))
                log(INFO, f"[make_postprocessing] area {area} has 0 train samples")
            elif tmp_X_train.ndim == 1:
                tmp_X_train = tmp_X_train.reshape((len(tmp_X_train), -1))
            if tmp_X_val.size == 0:
                tmp_X_val = tmp_X_val.reshape((0, total_cols))
                log(INFO, f"[make_postprocessing] area {area} has 0 val samples")
            elif tmp_X_val.ndim == 1:
                tmp_X_val = tmp_X_val.reshape((len(tmp_X_val), -1))

            area_X_train[area] = tmp_X_train
            area_X_val[area] = tmp_X_val
            area_y_train[area] = tmp_y_train
            area_y_val[area] = tmp_y_val

        # 删除 train 和 val 都为空的 area（不创建空客户端），并同步移除 exogenous 与 scaler key
        empty_areas = [a for a in list(area_X_train.keys()) if area_X_train[a].shape[0] == 0 and area_X_val[a].shape[0] == 0]
        for a in empty_areas:
            log(INFO, f"[make_postprocessing] Removing area {a} because both train and val are empty after preprocessing.")
            del area_X_train[a]
            del area_X_val[a]
            del area_y_train[a]
            del area_y_val[a]
            if area_exogenous_data_train is not None:
                area_exogenous_data_train.pop(a, None)
            if area_exogenous_data_val is not None:
                area_exogenous_data_val.pop(a, None)
            if a in y_scalers:
                y_scalers.pop(a)
    else:
        area_X_train, area_X_val, area_y_train, area_y_val = None, None, None, None
        area_exogenous_data_train, area_exogenous_data_val = None, None

    # transform to np
    if area_X_train is not None:
        for area in list(area_X_train.keys()):
            # 如果已经是 numpy（前面已转换），则跳过
            if isinstance(area_X_train[area], np.ndarray):
                continue
            tmp_X_train, tmp_y_train, tmp_X_val, tmp_y_val = remove_identifiers(
                area_X_train[area], area_y_train[area], area_X_val[area], area_y_val[area]
            )
            tmp_X_train = tmp_X_train.to_numpy()
            tmp_y_train = tmp_y_train.to_numpy()
            tmp_X_val = tmp_X_val.to_numpy()
            tmp_y_val = tmp_y_val.to_numpy()
            area_X_train[area] = tmp_X_train
            area_X_val[area] = tmp_X_val
            area_y_train[area] = tmp_y_train
            area_y_val[area] = tmp_y_val

    # 这里直接用 area_exogenous_data_train/val，不要再 to_numpy
    exogenous_data_train = area_exogenous_data_train
    exogenous_data_val = area_exogenous_data_val

    # remove identifiers from features, targets
    X_train_with_id = X_train
    X_val_with_id = X_val
    X_train, y_train, X_val, y_val = remove_identifiers(X_train, y_train, X_val, y_val)


    def _base_col_name(col: str) -> str:
        # generate_time_lags 命名格式：xxx_lag-1
        if "_lag-" in col:
            return col.split("_lag-")[0]
        return col

    def _pick_base_column(columns, base_name: str) -> str:
        # 优先使用 lag-1 作为代表列
        preferred = f"{base_name}_lag-1"
        if preferred in columns:
            return preferred
        for c in columns:
            if _base_col_name(c) == base_name:
                return c
        return ""

    def _compute_feature_stats_from_clients(per_area_arrays, columns, base_features):
        if per_area_arrays is None or columns is None:
            return {}
        col_index = {c: i for i, c in enumerate(columns)}
        stats = {}
        for b in base_features:
            col = _pick_base_column(columns, b)
            if not col or col not in col_index:
                continue
            col_idx = col_index[col]
            means, stds = [], []
            for _, arr in per_area_arrays.items():
                if arr is None or arr.size == 0 or arr.shape[1] <= col_idx:
                    continue
                vals = arr[:, col_idx]
                if vals.size == 0:
                    continue
                means.append(float(np.nanmean(vals)))
                if vals.size >= 2:
                    stds.append(float(np.nanstd(vals, ddof=1)))
                else:
                    stds.append(0.0)
            if means:
                stats[b] = {"means": means, "stds": stds}
        return stats

    def _infer_groups_by_variance(X_train_with_id, base_features, per_area_feature_stats=None, per_area_arrays=None, columns=None):
        # 数据驱动：按“组内方差/组间方差”比值划分动态/静态
        # 仅使用每个客户端的聚合统计（mean/std），可在联邦场景中由客户端本地计算后汇总。
        if (per_area_feature_stats is None and (per_area_arrays is None or columns is None)) and (X_train_with_id is None or args.identifier not in X_train_with_id.columns):
            return None, None, {}

        ratio_threshold = float(getattr(args, "group_split_within_ratio", 1.0))
        dynamic_bases, static_bases = [], []
        stats = {}
        col_index = {c: i for i, c in enumerate(columns)} if columns is not None else {}
        col_source = list(X_train_with_id.columns) if X_train_with_id is not None else list(columns or [])
        for b in base_features:
            col = _pick_base_column(col_source, b)
            if not col:
                continue

            if per_area_feature_stats is not None and b in per_area_feature_stats:
                means = per_area_feature_stats[b].get("means", [])
                stds = per_area_feature_stats[b].get("stds", [])
                if len(means) < 2:
                    continue
                within = float(np.mean(stds)) if len(stds) > 0 else 0.0
                between = float(np.std(means, ddof=0)) if len(means) > 0 else 0.0
            elif per_area_arrays is not None and columns is not None and col in col_index:
                col_idx = col_index[col]
                means, stds = [], []
                for _, arr in per_area_arrays.items():
                    if arr is None or arr.size == 0 or arr.shape[1] <= col_idx:
                        continue
                    vals = arr[:, col_idx]
                    if vals.size == 0:
                        continue
                    means.append(float(np.nanmean(vals)))
                    if vals.size >= 2:
                        stds.append(float(np.nanstd(vals, ddof=1)))
                    else:
                        stds.append(0.0)
                if len(means) < 2:
                    continue
                within = float(np.mean(stds)) if len(stds) > 0 else 0.0
                between = float(np.std(means, ddof=0)) if len(means) > 0 else 0.0
            else:
                tmp = X_train_with_id[[args.identifier, col]].dropna()
                if tmp[args.identifier].nunique() < 2:
                    continue
                per_area_stats = tmp.groupby(args.identifier)[col].agg(["mean", "std"])
                within = float(per_area_stats["std"].mean()) if len(per_area_stats) > 0 else 0.0
                between = float(per_area_stats["mean"].std(ddof=0)) if len(per_area_stats) > 0 else 0.0

            ratio = within / (between + 1e-8)
            stats[b] = {"within": within, "between": between, "ratio": ratio}

            if ratio >= ratio_threshold:
                dynamic_bases.append(b)
            else:
                static_bases.append(b)

        if len(dynamic_bases) == 0 or len(static_bases) == 0:
            return None, None, stats
        return set(dynamic_bases), set(static_bases), stats

    def build_group_feature_splits_from_columns(columns, X_train_with_id=None, per_area_feature_stats=None, per_area_arrays=None):
        # 固定 2 组：动态时变 / 静态建筑
        # 注意：返回的索引必须对应时序输入最后一维（基础特征维），不能是 lag 展开后的列索引。
        dynamic_set = {"hour", "day_of_week", "month", "air_temperature", "dew_temperature", "wind_speed"}
        static_set = {"square_feet", "age", "primary_use_code", "meter", "building_id", "site_id"}

        groups = {"dynamic": [], "static": []}
        base_features = []
        for c in columns:
            b = _base_col_name(c)
            if b not in base_features:
                base_features.append(b)

        dynamic_bases, static_bases, split_stats = _infer_groups_by_variance(
            X_train_with_id,
            base_features,
            per_area_feature_stats=per_area_feature_stats,
            per_area_arrays=per_area_arrays,
            columns=columns,
        )
        if dynamic_bases is not None:
            log(INFO, f"group split by variance: dynamic={sorted(dynamic_bases)}, static={sorted(static_bases)}")
        else:
            log(INFO, "group split by variance not available, fallback to prior sets")

        for i, b in enumerate(base_features):
            if dynamic_bases is not None:
                if b in dynamic_bases:
                    groups["dynamic"].append(i)
                elif b in static_bases:
                    groups["static"].append(i)
                else:
                    # 未分配特征回退到先验集合
                    if b in dynamic_set:
                        groups["dynamic"].append(i)
                    else:
                        groups["static"].append(i)
            else:
                if b in dynamic_set:
                    groups["dynamic"].append(i)
                elif b in static_set:
                    groups["static"].append(i)
                else:
                    groups["static"].append(i)

        # 可选：两组共享少量关键桥接特征。
        # 原始 dynamic/static 是互斥划分；开启后仍保持 2 个组，但允许关键特征同时进入两组。
        # 默认 auto 模式继续使用客户端统计汇总出的 within/between/ratio 自动挑选共享特征：
        # - dynamic -> static: 选择 ratio 最靠近动静态边界的动态特征，给静态组补充弱动态桥接信息；
        # - static -> dynamic: 选择 between 最大的静态特征，给动态组补充最能区分客户端的建筑上下文。
        overlap_enabled = bool(getattr(args, "group_overlap_enabled", True))
        if overlap_enabled and "dynamic" in groups and "static" in groups:
            overlap_mode = str(getattr(args, "group_overlap_mode", "auto")).lower()
            max_per_direction = int(getattr(args, "group_overlap_max_per_direction", 2))
            max_dynamic_to_static = int(getattr(
                args,
                "group_overlap_max_dynamic_to_static",
                max_per_direction,
            ))
            max_static_to_dynamic = int(getattr(
                args,
                "group_overlap_max_static_to_dynamic",
                max_per_direction,
            ))
            overlap_scheme_name = str(getattr(
                args,
                "group_overlap_scheme_name",
                f"two_group_{overlap_mode}_overlap_d2s{max_dynamic_to_static}_s2d{max_static_to_dynamic}",
            ))

            def _append_unique(group_name, feat_idx):
                if feat_idx not in groups[group_name]:
                    groups[group_name].append(feat_idx)

            added_dynamic_to_static, added_static_to_dynamic = [], []

            if overlap_mode == "auto":
                ratio_threshold = float(getattr(args, "group_split_within_ratio", 1.0))

                def _stat_value(base_name, key, default=0.0):
                    value = split_stats.get(base_name, {}).get(key, default)
                    try:
                        value = float(value)
                    except Exception:
                        value = default
                    if not np.isfinite(value):
                        value = default
                    return value

                dynamic_candidates = []
                static_candidates = []
                unscaled_stats = getattr(args, "group_unscaled_feature_stats", {})
                for i, b in enumerate(base_features):
                    if i in groups["dynamic"]:
                        ratio = _stat_value(b, "ratio", np.inf)
                        dynamic_candidates.append((abs(ratio - ratio_threshold), ratio, i, b))
                    if i in groups["static"]:
                        ratio = _stat_value(b, "ratio", 0.0)
                        raw_stat = unscaled_stats.get(b, {})
                        between = raw_stat.get("between", split_stats.get(b, {}).get("between", 0.0))
                        try:
                            between = float(between)
                        except Exception:
                            between = 0.0
                        if not np.isfinite(between):
                            between = 0.0
                        static_candidates.append((-between, -ratio, i, b))

                dynamic_candidates.sort(key=lambda x: (x[0], x[1], x[3]))
                static_candidates.sort(key=lambda x: (x[0], x[1], x[3]))

                for _, ratio, i, b in dynamic_candidates[:max_dynamic_to_static]:
                    _append_unique("static", i)
                    added_dynamic_to_static.append(f"{b}(ratio={ratio:.4g})")

                for neg_between, neg_ratio, i, b in static_candidates[:max_static_to_dynamic]:
                    _append_unique("dynamic", i)
                    added_static_to_dynamic.append(
                        f"{b}(between={-neg_between:.4g}, ratio={-neg_ratio:.4g})"
                    )
            elif overlap_mode == "manual":
                dynamic_to_static = set(getattr(
                    args,
                    "group_overlap_dynamic_to_static",
                    ["meter_reading", "hour", "sin_hour", "cos_hour"]
                ))
                static_to_dynamic = set(getattr(
                    args,
                    "group_overlap_static_to_dynamic",
                    ["primary_use_code", "meter", "square_feet"]
                ))
                for i, b in enumerate(base_features):
                    if b in dynamic_to_static and i in groups["dynamic"] and len(added_dynamic_to_static) < max_dynamic_to_static:
                        _append_unique("static", i)
                        added_dynamic_to_static.append(b)
                    if b in static_to_dynamic and i in groups["static"] and len(added_static_to_dynamic) < max_static_to_dynamic:
                        _append_unique("dynamic", i)
                        added_static_to_dynamic.append(b)
            else:
                raise ValueError(f"Unsupported group_overlap_mode={overlap_mode}. Use 'auto' or 'manual'.")

            log(
                INFO,
                f"group overlap enabled ({overlap_mode}, scheme={overlap_scheme_name}, "
                f"max_dynamic_to_static={max_dynamic_to_static}, "
                f"max_static_to_dynamic={max_static_to_dynamic}): "
                f"dynamic->static={added_dynamic_to_static}, "
                f"static->dynamic={added_static_to_dynamic}"
            )

        # 防止空组
        groups = {k: v for k, v in groups.items() if len(v) > 0}
        if len(base_features) > 0 and len(groups) > 0:
            max_idx = max(max(v) for v in groups.values())
            log(INFO, f"group split built on base features: n_base={len(base_features)}, max_idx={max_idx}")
        return groups

    assert len(X_train.columns) == len(X_val.columns)

    num_features = len(X_train.columns) // args.num_lags


    feature_cols = list(X_train.columns)
    base_feature_names = []
    for c in feature_cols:
        b = _base_col_name(c)
        if b not in base_feature_names:
            base_feature_names.append(b)
    per_area_feature_stats = _compute_feature_stats_from_clients(area_X_train, feature_cols, base_feature_names)
    args.group_feature_splits = build_group_feature_splits_from_columns(
        feature_cols,
        X_train_with_id=X_train_with_id,
        per_area_feature_stats=per_area_feature_stats,
        per_area_arrays=None,
    )
    args.num_groups = len(args.group_feature_splits)
    max_group_idx = max(max(v) for v in args.group_feature_splits.values())
    if max_group_idx >= num_features:
        raise ValueError(
            f"group_feature_splits 索引越界: max_idx={max_group_idx}, num_features={num_features}"
        )
    log(INFO, f"group_feature_splits: {args.group_feature_splits}")
    split_name_map = {
        g: [base_feature_names[i] for i in idxs if 0 <= i < len(base_feature_names)]
        for g, idxs in args.group_feature_splits.items()
    }
    log(INFO, f"group_feature_names: {split_name_map}")
    log(
        INFO,
        f"group_mode config: num_groups={args.num_groups}, experts_per_group={args.num_experts}, "
        f"total={args.num_groups * args.num_experts}, "
        f"group_use_full_features={bool(getattr(args, 'group_use_full_features', True))}"
    )


    # to timeseries representation
    X_train = to_timeseries_rep(X_train.to_numpy(), num_lags=args.num_lags,
                                num_features=num_features)
    X_val = to_timeseries_rep(X_val.to_numpy(), num_lags=args.num_lags,
                              num_features=num_features)

    if area_X_train is not None:
        area_X_train = to_timeseries_rep(area_X_train, num_lags=args.num_lags,
                                         num_features=num_features)
        area_X_val = to_timeseries_rep(area_X_val, num_lags=args.num_lags,
                                       num_features=num_features)

    # transform targets to numpy
    y_train, y_val = y_train.to_numpy(), y_val.to_numpy()

    if exogenous_data_train is not None:
        exogenous_data_train_combined, exogenous_data_val_combined = [], []
        for area in exogenous_data_train:
            arr_train = np.asarray(exogenous_data_train[area])
            arr_val = np.asarray(exogenous_data_val[area])
            if arr_train.ndim == 1:
                arr_train = arr_train.reshape(1, -1)
            if arr_val.ndim == 1:
                arr_val = arr_val.reshape(1, -1)
            exogenous_data_train_combined.append(arr_train)
            exogenous_data_val_combined.append(arr_val)
        # 仅在存在外生数据时合并 "all"
        exogenous_data_train["all"] = np.vstack(exogenous_data_train_combined)
        exogenous_data_val["all"] = np.vstack(exogenous_data_val_combined)

    # 新增：根据生成的 area keys 更新 y_scalers 的键（兼容 'area' 或 'area_Client_x'）
    if area_X_train is not None:
        client_keys = list(area_X_train.keys())
        y_scalers = update_scaler_keys(y_scalers, client_keys)

    return X_train, X_val, y_train, y_val, area_X_train, area_X_val, area_y_train, area_y_val, exogenous_data_train, exogenous_data_val, y_scalers


X_train, X_val, y_train, y_val, client_X_train, client_X_val, client_y_train, client_y_val, exogenous_data_train, exogenous_data_val, y_scalers = make_postprocessing(
    X_train, X_val, y_train, y_val, exogenous_data_train, exogenous_data_val, x_scalers, y_scalers
)

# 打印所有 ElBorn 子客户端的键和数据形状（仅在外生数据存在时）
if exogenous_data_train is not None:
    elborn_clients = [key for key in exogenous_data_train.keys() if "ElBorn" in key]
    for client in elborn_clients:
        print(f"Client: {client}, Data Shape: {exogenous_data_train[client].shape}")

# 计算某些统计信息（仅在外生数据存在时）
if exogenous_data_train is not None:
    for client in elborn_clients:
        print(f"Client: {client}, Feature Length: {len(exogenous_data_train[client][0])}")


def get_input_dims(X_train, exogenous_data_train):
    if args.model_name == "mlp":
        input_dim = X_train.shape[1] * X_train.shape[2]
    else:
        input_dim = X_train.shape[2]

    if exogenous_data_train is not None:
        if len(exogenous_data_train) == 1:
            cid = next(iter(exogenous_data_train.keys()))
            exogenous_dim = exogenous_data_train[cid].shape[1]
        else:
            exogenous_dim = exogenous_data_train["all"].shape[1]
    else:
        exogenous_dim = 0

    return input_dim, exogenous_dim


def collect_selected_experts_from_server(server):
    """
    从 server 的 client_proxies 中收集每个客户端最终用于推理的专家编号。
    优先级：
    1. 训练后 DQN 网络对当前状态的一次贪心策略输出
    2. 若失败，再回退到 selected_expert
    """
    selected_experts = {}

    for cp in getattr(server, "client_proxies", []):
        cid = getattr(cp, "cid", None)
        client_obj = getattr(cp, "client", cp)

        selected_expert = None

        selector = getattr(client_obj, "selector", None)
        collect_state_fn = getattr(client_obj, "_collect_state", None)

        if selector is not None and callable(collect_state_fn):
            try:
                state = collect_state_fn()
                state_tensor = torch.tensor(
                    state, dtype=torch.float32, device=selector.device
                ).unsqueeze(0)
                if hasattr(selector, "select_greedy"):
                    selected_expert = selector.select_greedy(state)
                else:
                    with torch.no_grad():
                        selected_expert = int(selector.net(state_tensor).argmax().item())
                if isinstance(selected_expert, np.ndarray):
                    selected_expert = selected_expert.astype(np.int64).tolist()
                log(INFO, f"[{cid}] use greedy DQN policy output: {selected_expert}")
            except Exception as e:
                log(INFO, f"[{cid}] failed to get greedy DQN policy output: {e}")
                selected_expert = None

        if selected_expert is None:
            selected_expert = getattr(cp, "selected_expert", None)
        if selected_expert is None:
            selected_expert = getattr(client_obj, "selected_expert", None)

        selected_experts[cid] = selected_expert

    return selected_experts



input_dim, exogenous_dim = get_input_dims(X_train, exogenous_data_train)

print(input_dim, exogenous_dim)

model = get_model(model=args.model_name,
                  input_dim=input_dim,
                  out_dim=y_train.shape[1],
                  lags=args.num_lags,
                  exogenous_dim=exogenous_dim,
                  seed=args.seed)

print(model)


def fit(model, X_train, y_train, X_val, y_val,
        exogenous_data_train=None, exogenous_data_val=None,
        idxs=[0],  # the indices of our targets in X
        log_per=1,
        client_creation_fn=None,  # client specification
        local_train_params=None,  # local params
        aggregation_params=None,  # aggregation params
        use_carbontracker=True):
    # client creation definition
    if client_creation_fn is None:
        client_creation_fn = create_regression_client
    # local params
    if local_train_params is None:
        local_train_params = {
            "epochs": args.epochs, "optimizer": args.optimizer, "lr": args.lr,
            "criterion": args.criterion, "early_stopping": args.local_early_stopping,
            "patience": args.local_patience, "device": device
        }

    train_loaders, val_loaders = [], []

    # get data per client
    for client in X_train:
        if client == "all":
            continue
        if exogenous_data_train is not None:
            tmp_exogenous_data_train = exogenous_data_train.get(client, None)
            tmp_exogenous_data_val = exogenous_data_val.get(client, None) if exogenous_data_val is not None else None
            # convert DataFrame -> numpy and ensure 2D (samples, features)
            if isinstance(tmp_exogenous_data_train, pd.DataFrame):
                tmp_exogenous_data_train = tmp_exogenous_data_train.to_numpy()
            if isinstance(tmp_exogenous_data_val, pd.DataFrame):
                tmp_exogenous_data_val = tmp_exogenous_data_val.to_numpy()
            if tmp_exogenous_data_train is not None:
                tmp_exogenous_data_train = np.asarray(tmp_exogenous_data_train)
                if tmp_exogenous_data_train.ndim == 1:
                    tmp_exogenous_data_train = tmp_exogenous_data_train.reshape(-1, 1)
            if tmp_exogenous_data_val is not None:
                tmp_exogenous_data_val = np.asarray(tmp_exogenous_data_val)
                if tmp_exogenous_data_val.ndim == 1:
                    tmp_exogenous_data_val = tmp_exogenous_data_val.reshape(-1, 1)
        else:
            tmp_exogenous_data_train = None
            tmp_exogenous_data_val = None

        print(f"[{client}] X: {X_train[client].shape}, y: {y_train[client].shape}, exogenous: {tmp_exogenous_data_train.shape if tmp_exogenous_data_train is not None else None}")
        num_features = len(X_train[client][0][0])

        # to torch loader
        train_loaders.append(
            to_torch_dataset(
                X_train[client], y_train[client],
                num_lags=args.num_lags,
                num_features=num_features,
                exogenous_data=tmp_exogenous_data_train,
                indices=idxs,
                batch_size=args.batch_size,
                shuffle=False
            )
        )
        val_loaders.append(
            to_torch_dataset(
                X_val[client], y_val[client],
                num_lags=args.num_lags,
                exogenous_data=tmp_exogenous_data_val,
                indices=idxs,
                batch_size=args.batch_size,
                shuffle=False
            )
        )

    cids = [k for k in X_train.keys() if k != "all"]
    clients = [
        client_creation_fn(
            cid=cid,
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            local_params=local_train_params
        )
        for cid, train_loader, val_loader in zip(cids, train_loaders, val_loaders)
    ]

    # represent clients to server
    client_proxies = [
        SimpleClientProxy(cid, client) for cid, client in zip(cids, clients)
    ]

    # represent the server
    server = Server(
        X_train=X_train,
        exogenous_data_train=exogenous_data_train,
        client_proxies=client_proxies,
        aggregation=args.aggregation,
        aggregation_params=aggregation_params,
        local_params_fn=None,
    )

    def _diagnose_server_clients(server):
        import torch
        import traceback

        issues = []
        for cp in getattr(server, "client_proxies", []):
            # support SimpleClientProxy wrapper
            client_obj = getattr(cp, "client", cp)
            train_loader = getattr(client_obj, "train_loader", None)
            net = getattr(client_obj, "net", getattr(cp, "net", None))
            cid = getattr(cp, "cid", getattr(client_obj, "cid", "unknown"))
            try:
                if train_loader is None:
                    print(f"[{cid}] NO train_loader on proxy/client object")
                    issues.append(cid)
                    continue
                xb, exb, y_hist, yb = next(iter(train_loader))
                x_last = xb[:, -1, :].view(xb.size(0), -1)
                ex_shape = None
                if exb is not None:
                    ex_shape = tuple(exb[:, -1, :].shape) if (hasattr(exb, "dim") and exb.dim() == 3) else tuple(exb.shape)
                # gate first linear expected dim
                gate_in = None
                gate_layer = getattr(net, "gate", None)
                if isinstance(gate_layer, torch.nn.Linear):
                    gate_in = gate_layer.in_features
                elif gate_layer is not None:
                    try:
                        for m in gate_layer:
                            if isinstance(m, torch.nn.Linear):
                                gate_in = m.in_features
                                break
                    except TypeError:
                        gate_in = None
                fc_in = getattr(net, "fc_in_dim", None)
                print(f"[{cid}] x_last={tuple(x_last.shape)}, exog={ex_shape}, gate_in={gate_in}, fc_in={fc_in}, net_exog={getattr(net, 'exogenous_dim', None)}")
                # show constructed gating_input shape (what forward would build)
                try:
                    # emulate forward combine logic
                    combined = x_last
                    if exb is not None:
                        ex_last = exb[:, -1, :] if exb.dim() == 3 else exb
                        combined = torch.cat([x_last, ex_last], dim=1)
                    print(f"  constructed combined.shape={tuple(combined.shape)}")
                except Exception as e:
                    print(f"  combined build failed: {e}")
                # try one forward to see exact error
                try:
                    with torch.no_grad():
                        net = net.to(device)
                        if hasattr(net, "gate"):
                            out = net(xb.to(device), exb.to(device) if exb is not None else None, device=device, return_gates=True)
                            if isinstance(out, tuple):
                                _, gates = out
                                mean_gates = gates.mean(dim=0)
                                print(f"  gate mean probs sample: {mean_gates.cpu().numpy().tolist()}")
                        else:
                            out = net(xb.to(device), exb.to(device) if exb is not None else None, device=device)
                    print("  forward OK")
                except Exception as e:
                    print("  forward ERROR:", e)
                    traceback.print_exc()
                    issues.append(cid)
            except Exception as e:
                print(f"[{cid}] sample fetch failed: {e}")
                traceback.print_exc()
                issues.append(cid)
        if issues:
            print("Diagnostics: problems found in clients:", issues)
        else:
            print("Diagnostics: all clients OK.")

    _diagnose_server_clients(server)

    global_model, history = server.fit(args.fl_rounds, args.fraction, use_carbontracker=use_carbontracker)

    selected_experts_by_client = getattr(global_model, "selected_experts_by_client", None)

    if not selected_experts_by_client:
        selected_experts_by_client = getattr(server, "best_selected_experts_by_client", {})

    if not selected_experts_by_client:
        log(INFO, "[Inference Prep] best round greedy expert map missing, fallback to final server state collection")
        selected_experts_by_client = collect_selected_experts_from_server(server)

    global_model.selected_experts_by_client = selected_experts_by_client
    log(INFO, f"[Inference Prep] selected_experts_by_client(best-aligned): {selected_experts_by_client}")

    return global_model, history


# federated local params
local_train_params = {"epochs": args.epochs, "optimizer": args.optimizer, "lr": args.lr,
                      "criterion": args.criterion, "early_stopping": args.local_early_stopping,
                      "patience": args.local_patience, "device": device
                      }

global_model, history = fit(
    model,
    client_X_train,
    client_y_train,
    client_X_val,
    client_y_val,
    local_train_params=local_train_params,
    exogenous_data_train=exogenous_data_train,
    exogenous_data_val=exogenous_data_val
)


def transform_preds(y_pred_train, y_pred_val):
    if not isinstance(y_pred_train, np.ndarray):
        y_pred_train = y_pred_train.cpu().numpy()
    if not isinstance(y_pred_val, np.ndarray):
        y_pred_val = y_pred_val.cpu().numpy()
    return y_pred_train, y_pred_val


def round_predictions(y_pred_train, y_pred_val, dims):
    # round to closest integer
    if dims is None or len(dims) == 0:
        return y_pred_train, y_pred_val
    for dim in dims:
        y_pred_train[:, dim] = np.rint(y_pred_train[:, dim])
        y_pred_val[:, dim] = np.rint(y_pred_val[:, dim])
    return y_pred_train, y_pred_val


def inverse_transform(y_train, y_val, y_pred_train, y_pred_val,
                      y_scaler=None,
                      round_preds=False, dims=None):
    y_pred_train, y_pred_val = transform_preds(y_pred_train, y_pred_val)

    if y_scaler is not None:
        y_train = y_scaler.inverse_transform(y_train)
        y_val = y_scaler.inverse_transform(y_val)
        y_pred_train = y_scaler.inverse_transform(y_pred_train)
        y_pred_val = y_scaler.inverse_transform(y_pred_val)

    # to zeroes
    y_pred_train[y_pred_train < 0.] = 0.
    y_pred_val[y_pred_val < 0.] = 0.

    if round_preds:
        y_pred_train, y_pred_val = round_predictions(y_pred_train, y_pred_val, dims)

    return y_train, y_val, y_pred_train, y_pred_val


def make_plot(y_true, y_pred,
              title,
              feature_names=None,
              client=None,
              save_dir=None):
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(y_pred.shape[1])]
    assert len(feature_names) == y_pred.shape[1]

    print("画图用的y_true[:5]:", y_true[:5])
    print("画图用的y_pred[:5]:", y_pred[:5])

    if save_dir is None:
        save_dir = os.getenv("PLOTS_DIR", "./plots")
    # 创建保存目录（如果不存在）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(y_pred.shape[1]):
        plt.figure(figsize=(8, 6))
        plt.ticklabel_format(style='plain')
        plt.plot(y_true[:, i], label="Actual")
        plt.plot(y_pred[:, i], label="Predicted")
        if client is not None:
            plt.title(f"[{client} {title}] {feature_names[i]} prediction")
            save_path = os.path.join(save_dir, f"{client}_{title}_{feature_names[i]}.png")
        else:
            plt.title(f"[{title}] {feature_names[i]} prediction")
            save_path = os.path.join(save_dir, f"{title}_{feature_names[i]}.png")

        plt.legend()
        plt.savefig(save_path)
        plt.close()


def inference(
    model,  # the global model
    client_X_train,  # train data per client
    client_y_train,
    client_X_val,  # val data per client
    client_y_val,
    exogenous_data_train,  # exogenous data per client
    exogenous_data_val,
    original_y_scalers,  # 使用原始的 y_scalers
    idxs=[0],
    apply_round=False,  # 关闭四舍五入
    round_dimensions=[0],  # the dimensions to apply rounding
    plot=True,  # plot predictions
):
    import itertools
    import pandas as pd
    import os

    save_dir = os.getenv("RESULTS_DIR", "./results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    selected_experts_by_client = getattr(model, "selected_experts_by_client", {})
    log(INFO, f"[Inference] selected_experts_by_client: {selected_experts_by_client}")


    is_group_mode = getattr(model, "use_group_experts", False)  

    fallback_enabled = bool(getattr(args, "inference_group_fallback_enabled", True))
    fallback_r2_trigger = float(getattr(args, "inference_group_fallback_r2_trigger", -0.5))
    fallback_nrmse_trigger = float(getattr(args, "inference_group_fallback_nrmse_trigger", 2.0))
    fallback_max_candidates = int(getattr(args, "inference_group_fallback_max_candidates", 64))
    fallback_mse_trigger = float(getattr(args, "inference_group_fallback_mse_trigger", -1.0))
    fallback_val_train_mse_ratio_trigger = float(getattr(args, "inference_group_fallback_val_train_mse_ratio_trigger", -1.0))
    fallback_val_train_mse_min = float(getattr(args, "inference_group_fallback_val_train_mse_min", 0.0))
    fallback_min_r2_gain = float(getattr(args, "inference_group_fallback_min_r2_gain", 0.02))
    fallback_min_mse_rel_gain = float(getattr(args, "inference_group_fallback_min_mse_rel_gain", 0.03))
    fallback_selection_metric = str(getattr(args, "inference_group_fallback_selection_metric", "r2")).lower()
    fallback_min_raw_mse_rel_gain = float(getattr(args, "inference_group_fallback_min_raw_mse_rel_gain", 0.0))
    fallback_raw_mse_trigger = float(getattr(args, "inference_group_fallback_raw_mse_trigger", -1.0))

    def _normalize_group_selected_experts(raw_selected):
        if raw_selected is None:
            return None
        if isinstance(raw_selected, np.ndarray):
            raw_selected = raw_selected.astype(np.int64).tolist()
        elif isinstance(raw_selected, tuple):
            raw_selected = list(raw_selected)
        elif isinstance(raw_selected, (int, np.integer)):
            num_groups = int(getattr(model, "num_groups", 1))
            experts_per_group = int(getattr(model, "experts_per_group", getattr(model, "num_experts", 1)))
            flat = int(raw_selected)
            g = flat // experts_per_group
            e = flat % experts_per_group
            g = max(0, min(num_groups - 1, g))
            selected = [0] * num_groups
            selected[g] = e
            raw_selected = selected

        if not isinstance(raw_selected, list):
            return None
        num_groups = int(getattr(model, "num_groups", 1))
        experts_per_group = int(getattr(model, "experts_per_group", getattr(model, "num_experts", 1)))
        if len(raw_selected) != num_groups:
            return None
        norm = []
        for e in raw_selected:
            e_int = int(e)
            e_int = max(0, min(experts_per_group - 1, e_int))
            norm.append(e_int)
        return norm

    def _as_2d_array(values):
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()
        arr = np.asarray(values)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def _raw_scale_mse(cid, y_pred):
        area = cid.split("_Client_")[0] if "_Client_" in cid else cid
        y_scaler = original_y_scalers.get(area, None)
        if y_scaler is None:
            y_scaler = original_y_scalers.get(cid, None)
        if y_scaler is None:
            return None
        try:
            y_true = y_scaler.inverse_transform(_as_2d_array(client_y_val[cid]))
            y_hat = y_scaler.inverse_transform(_as_2d_array(y_pred))
            return float(np.mean((y_true - y_hat) ** 2))
        except Exception as exc:
            log(INFO, f"[{cid}] raw_mse unavailable: {exc}")
            return None

    def _fallback_key(r2, mse, raw_mse):
        if fallback_selection_metric in ("raw_mse", "inverse_mse", "real_mse") and raw_mse is not None:
            return (-raw_mse, r2, -mse)
        return (r2, -mse)

    def _search_group_fallback_experts(cid, val_loader, current_selected):
        num_groups = int(getattr(model, "num_groups", 1))
        experts_per_group = int(getattr(model, "experts_per_group", getattr(model, "num_experts", 1)))
        if num_groups <= 0 or experts_per_group <= 0:
            return None

        total_candidates = experts_per_group ** num_groups
        if total_candidates <= 0:
            return None

        all_candidates = list(itertools.product(range(experts_per_group), repeat=num_groups))
        candidates = all_candidates
        if fallback_max_candidates > 0 and len(candidates) > fallback_max_candidates:
            seed_offset = sum(ord(ch) for ch in str(cid))
            rng = np.random.default_rng(int(getattr(args, "seed", 42)) + seed_offset)
            keep_idx = set()
            if current_selected is not None:
                try:
                    keep_idx.add(all_candidates.index(tuple(current_selected)))
                except ValueError:
                    pass
            remaining = [i for i in range(len(all_candidates)) if i not in keep_idx]
            need = max(0, fallback_max_candidates - len(keep_idx))
            if need > 0 and len(remaining) > 0:
                sampled = rng.choice(remaining, size=min(need, len(remaining)), replace=False)
                for i in np.atleast_1d(sampled).tolist():
                    keep_idx.add(int(i))
            candidates = [all_candidates[i] for i in sorted(keep_idx)]

        best = None
        for cand in candidates:
            cand_list = [int(x) for x in cand]
            cand_mse, cand_rmse, cand_mae, cand_r2, cand_nrmse, cand_pred = test(
                model, val_loader, None, device=device, selected_experts=cand_list
            )
            cand_raw_mse = _raw_scale_mse(cid, cand_pred)
            key = _fallback_key(cand_r2, cand_mse, cand_raw_mse)
            if best is None or key > best["key"]:
                best = {
                    "selected_experts": cand_list,
                    "mse": cand_mse,
                    "rmse": cand_rmse,
                    "mae": cand_mae,
                    "r2": cand_r2,
                    "nrmse": cand_nrmse,
                    "raw_mse": cand_raw_mse,
                    "key": key,
                    "searched": len(candidates),
                    "total": total_candidates,
                }
        return best


    # load per client data to torch
    train_loaders, val_loaders = [], []

    # get data per client
    for client in client_X_train:
        if client == "all":
            continue
        if exogenous_data_train is not None:
            tmp_exogenous_data_train = exogenous_data_train[client]
            tmp_exogenous_data_val = exogenous_data_val[client]
        else:
            tmp_exogenous_data_train = None
            tmp_exogenous_data_val = None

        num_features = len(client_X_train[client][0][0])

        train_loaders.append(
            to_torch_dataset(
                client_X_train[client],
                client_y_train[client],
                num_lags=args.num_lags,
                num_features=num_features,
                exogenous_data=tmp_exogenous_data_train,
                indices=idxs,
                batch_size=1,
                shuffle=False,
            )
        )
        val_loaders.append(
            to_torch_dataset(
                client_X_val[client],
                client_y_val[client],
                num_lags=args.num_lags,
                exogenous_data=tmp_exogenous_data_val,
                indices=idxs,
                batch_size=1,
                shuffle=False,
            )
        )

    cids = [k for k in client_X_train.keys() if k != "all"]

    # 预测并逆归一化（每个客户端单独逆归一化）
    y_preds_train, y_preds_val = dict(), dict()
    y_train_inv, y_val_inv, y_pred_train_inv, y_pred_val_inv = dict(), dict(), dict(), dict()

    for cid, train_loader, val_loader in zip(cids, train_loaders, val_loaders):
        print(f"Prediction on {cid}")

        if is_group_mode:
            selected_experts = _normalize_group_selected_experts(selected_experts_by_client.get(cid, None))
            if selected_experts is not None:
                log(INFO, f"[{cid}] group-expert inference uses selected_experts={selected_experts}")
                train_mse, train_rmse, train_mae, train_r2, train_nrmse, y_pred_train = test(
                    model, train_loader, None, device=device, selected_experts=selected_experts
                )
                val_mse, val_rmse, val_mae, val_r2, val_nrmse, y_pred_val = test(
                    model, val_loader, None, device=device, selected_experts=selected_experts
                )
                val_raw_mse = _raw_scale_mse(cid, y_pred_val)

                val_train_mse_ratio = float(val_mse / (abs(train_mse) + 1e-8))
                mse_trigger = (fallback_mse_trigger > 0.0) and (val_mse >= fallback_mse_trigger)
                raw_mse_trigger = (
                    (fallback_raw_mse_trigger > 0.0)
                    and (val_raw_mse is not None)
                    and (val_raw_mse >= fallback_raw_mse_trigger)
                )
                mse_ratio_trigger = (
                    (fallback_val_train_mse_ratio_trigger > 0.0)
                    and (val_mse >= fallback_val_train_mse_min)
                    and (val_train_mse_ratio >= fallback_val_train_mse_ratio_trigger)
                )
                is_anomaly = (
                    (val_r2 <= fallback_r2_trigger)
                    or (val_nrmse >= fallback_nrmse_trigger)
                    or mse_trigger
                    or raw_mse_trigger
                    or mse_ratio_trigger
                )
                if fallback_enabled and is_anomaly:
                    fallback_best = _search_group_fallback_experts(cid, val_loader, selected_experts)
                    if fallback_best is not None:
                        r2_gain = float(fallback_best["r2"] - val_r2)
                        mse_rel_gain = float((val_mse - fallback_best["mse"]) / (abs(val_mse) + 1e-8))
                        if val_raw_mse is not None and fallback_best.get("raw_mse") is not None:
                            raw_mse_rel_gain = float(
                                (val_raw_mse - fallback_best["raw_mse"]) / (abs(val_raw_mse) + 1e-8)
                            )
                        else:
                            raw_mse_rel_gain = float("-inf")
                        metric_uses_raw_mse = fallback_selection_metric in ("raw_mse", "inverse_mse", "real_mse")
                        raw_mse_available = val_raw_mse is not None and fallback_best.get("raw_mse") is not None
                        if metric_uses_raw_mse and raw_mse_available:
                            should_switch = raw_mse_rel_gain >= fallback_min_raw_mse_rel_gain
                        else:
                            should_switch = (r2_gain >= fallback_min_r2_gain) or (mse_rel_gain >= fallback_min_mse_rel_gain)
                        reason_tags = []
                        if val_r2 <= fallback_r2_trigger:
                            reason_tags.append("r2")
                        if val_nrmse >= fallback_nrmse_trigger:
                            reason_tags.append("nrmse")
                        if mse_trigger:
                            reason_tags.append("mse")
                        if raw_mse_trigger:
                            reason_tags.append("raw_mse")
                        if mse_ratio_trigger:
                            reason_tags.append("mse_ratio")
                        log(INFO,
                            f"[{cid}] safeguard trigger={'+'.join(reason_tags) if reason_tags else 'none'}, "
                            f"val/train_mse_ratio={val_train_mse_ratio:.4f}, "
                            f"[{cid}] safeguard scan: searched={fallback_best['searched']}/{fallback_best['total']}, "
                            f"base(r2={val_r2:.4f}, mse={val_mse:.4f}, nrmse={val_nrmse:.4f}), "
                            f"best(r2={fallback_best['r2']:.4f}, mse={fallback_best['mse']:.4f}, "
                            f"nrmse={fallback_best['nrmse']:.4f}), "
                            f"gain(r2={r2_gain:.4f}, mse_rel={mse_rel_gain:.4f}, "
                            f"raw_mse_rel={raw_mse_rel_gain:.4f}), "
                            f"thresholds(r2={fallback_min_r2_gain:.4f}, "
                            f"mse_rel={fallback_min_mse_rel_gain:.4f}, "
                            f"raw_mse_rel={fallback_min_raw_mse_rel_gain:.4f}), "
                            f"raw_mse(base={val_raw_mse}, best={fallback_best.get('raw_mse')}), "
                            f"switch={should_switch}")
                        if should_switch:
                            selected_experts = fallback_best["selected_experts"]
                            selected_experts_by_client[cid] = selected_experts
                            model.selected_experts_by_client = selected_experts_by_client
                            log(INFO, f"[{cid}] safeguard switched selected_experts to {selected_experts}")
                            train_mse, train_rmse, train_mae, train_r2, train_nrmse, y_pred_train = test(
                                model, train_loader, None, device=device, selected_experts=selected_experts
                            )
                            val_mse, val_rmse, val_mae, val_r2, val_nrmse, y_pred_val = test(
                                model, val_loader, None, device=device, selected_experts=selected_experts
                            )
            else:
                log(INFO, f"[{cid}] group-expert inference fallback to soft mixture (no saved selected_experts)")
                train_mse, train_rmse, train_mae, train_r2, train_nrmse, y_pred_train = test(
                    model, train_loader, None, device=device
                )
                val_mse, val_rmse, val_mae, val_r2, val_nrmse, y_pred_val = test(
                    model, val_loader, None, device=device
                )
        else:
            selected_expert = selected_experts_by_client.get(cid, None)
            if selected_expert is not None:
                log(INFO, f"[{cid}] inference uses selected_expert={selected_expert}")
                train_mse, train_rmse, train_mae, train_r2, train_nrmse, y_pred_train = test(
                    model, train_loader, None, device=device, expert_idx=selected_expert
                )
                val_mse, val_rmse, val_mae, val_r2, val_nrmse, y_pred_val = test(
                    model, val_loader, None, device=device, expert_idx=selected_expert
                )
            else:
                log(INFO, f"[{cid}] no saved selected_expert found, fallback to mixture output")
                train_mse, train_rmse, train_mae, train_r2, train_nrmse, y_pred_train = test(
                    model, train_loader, None, device=device
                )
                val_mse, val_rmse, val_mae, val_r2, val_nrmse, y_pred_val = test(
                    model, val_loader, None, device=device
                )

        # 这段必须在 if/else 外层，保证两种模式都回填
        y_preds_train[cid] = y_pred_train
        y_preds_val[cid] = y_pred_val

        area = cid.split("_Client_")[0] if "_Client_" in cid else cid
        y_scaler = original_y_scalers[area]
        y_train, y_val = client_y_train[cid], client_y_val[cid]
        y_train, y_val, y_pred_train, y_pred_val = inverse_transform(
            y_train, y_val, y_pred_train, y_pred_val,
            y_scaler, round_preds=apply_round, dims=round_dimensions
        )
        y_train_inv[cid] = y_train
        y_val_inv[cid] = y_val
        y_pred_train_inv[cid] = y_pred_train
        y_pred_val_inv[cid] = y_pred_val

    summary_rows = []
    train_metrics = []
    val_metrics = []
    
    areas = sorted({cid.split("_Client_")[0] if "_Client_" in cid else cid for cid in cids})
    for area in areas:
        area_label = area
        cids_this_area = [cid for cid in cids if (cid.split("_Client_")[0] if "_Client_" in cid else cid) == area]
        # 每个 area 现在通常只有一个 cid，但仍沿用 concat 逻辑以兼容可能的多个子客户端
        y_train = np.concatenate([y_train_inv[cid] for cid in cids_this_area], axis=0)
        y_val = np.concatenate([y_val_inv[cid] for cid in cids_this_area], axis=0)
        y_pred_train = np.concatenate([y_pred_train_inv[cid] for cid in cids_this_area], axis=0)
        y_pred_val = np.concatenate([y_pred_val_inv[cid] for cid in cids_this_area], axis=0)

        # selected_experts_this_area = [
        #     selected_experts_by_client.get(cid, None) for cid in cids_this_area
        # ]
        if is_group_mode:
            selected_experts_this_area = [
                selected_experts_by_client.get(cid, None) for cid in cids_this_area
            ]
        else:
            selected_experts_this_area = [
                selected_experts_by_client.get(cid, None) for cid in cids_this_area
            ]
        print("After inverse_transform:")
        print("y_train[:5]:", y_train[:5])
        print("y_pred_train[:5]:", y_pred_train[:5])

        # 保存真实值和预测值到csv
        df_result_train = pd.DataFrame({
            "y_true": y_train.flatten(),
            "y_pred": y_pred_train.flatten()
        })
        df_result_train.to_csv(os.path.join(save_dir, f"{area_label}_train_pred.csv"), index=False)

        df_result_val = pd.DataFrame({
            "y_true": y_val.flatten(),
            "y_pred": y_pred_val.flatten()
        })
        df_result_val.to_csv(os.path.join(save_dir, f"{area_label}_val_pred.csv"), index=False)

        # 计算指标
        train_mse, train_rmse, train_mae, train_r2, train_nrmse, train_res_per_dim = accumulate_metric(
            y_train, y_pred_train, True, return_all=True
        )
        val_mse, val_rmse, val_mae, val_r2, val_nrmse, val_res_per_dim = accumulate_metric(
            y_val, y_pred_val, True, return_all=True
        )
        train_metrics.append([train_mse, train_rmse, train_mae, train_r2, train_nrmse])
        val_metrics.append([val_mse, val_rmse, val_mae, val_r2, val_nrmse])

        summary_rows.append({
            "area": area_label,
            "selected_expert": str(selected_experts_this_area),
            "train_mse": train_mse,
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "train_r2": train_r2,
            "train_nrmse": train_nrmse,
            "val_mse": val_mse,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "val_r2": val_r2,
            "val_nrmse": val_nrmse,
        })

        log(INFO, f"\nFinal Prediction on {area} (Inference Stage)")
        log(INFO, f"[{area}] selected_expert(s): {selected_experts_this_area}")
        log(INFO, f"[Train]: mse: {train_mse}, "
                  f"rmse: {train_rmse}, mae {train_mae}, r2: {train_r2}, nrmse: {train_nrmse}")
        log(INFO, f"[Val]: mse: {val_mse}, "
                  f"rmse: {val_rmse}, mae {val_mae}, r2: {val_r2}, nrmse: {val_nrmse}\n\n")

        if plot:
            make_plot(
                y_train,
                y_pred_train,
                title="Train",
                feature_names=args.targets,
                client=area_label,
                save_dir=os.getenv("PLOTS_DIR", "./plots"),
            )
            make_plot(
                y_val,
                y_pred_val,
                title="Val",
                feature_names=args.targets,
                client=area_label,
                save_dir=os.getenv("PLOTS_DIR", "./plots"),
            )

    # 转为numpy方便计算
    train_metrics = np.array(train_metrics)
    val_metrics = np.array(val_metrics)

    # 计算平均
    avg_train_metrics = train_metrics.mean(axis=0)
    avg_val_metrics = val_metrics.mean(axis=0)

    log(INFO, "\n=== 全局平均指标 (所有 areas 平均) ===")
    log(INFO, f"[Train] mse: {avg_train_metrics[0]}, rmse: {avg_train_metrics[1]}, mae: {avg_train_metrics[2]}, r2: {avg_train_metrics[3]}, nrmse: {avg_train_metrics[4]}")
    log(INFO, f"[Val]   mse: {avg_val_metrics[0]}, rmse: {avg_val_metrics[1]}, mae: {avg_val_metrics[2]}, r2: {avg_val_metrics[3]}, nrmse: {avg_val_metrics[4]}")

    # 保存全局平均指标
    summary_rows.append({
        "area": "average",
        "selected_expert": "",
        "train_mse": avg_train_metrics[0],
        "train_rmse": avg_train_metrics[1],
        "train_mae": avg_train_metrics[2],
        "train_r2": avg_train_metrics[3],
        "train_nrmse": avg_train_metrics[4],
        "val_mse": avg_val_metrics[0],
        "val_rmse": avg_val_metrics[1],
        "val_mae": avg_val_metrics[2],
        "val_r2": avg_val_metrics[3],
        "val_nrmse": avg_val_metrics[4],
    })
    pd.DataFrame(summary_rows).to_csv(os.path.join(save_dir, "result.csv"), index=False)


inference(
    global_model,
    client_X_train,
    client_y_train,
    client_X_val,
    client_y_val,
    exogenous_data_train,
    exogenous_data_val,
    original_y_scalers
)
