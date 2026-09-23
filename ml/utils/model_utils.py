from ml.models.rnn import RNN
from ml.models.lstm import LSTM
from ml.models.gru import GRU
from ml.models.cnn import CNN
from ml.models.rnn_autoencoder import DualAttentionAutoEncoder

from ml.models.moe_lstm import MoELSTM  # 新增导入

from argparse import Namespace
from pathlib import Path

from ml.models.group_moe_lstm import GroupMoELSTM

ROOT_DIR = Path(__file__).resolve().parents[2]

args = Namespace(

    #新增
    data_path=str(ROOT_DIR / 'dataset/十五个建筑2.csv'), # dataset

    # data_path='../dataset/full_dataset.csv', # dataset

    test_size=0.2, # validation size 


    #新增
    target='meter_reading', # the target column to predict
    # targets=['rnti_count', 'rb_down', 'rb_up', 'down', 'up'], # the target columns

    num_lags=24, # the number of past observations to feed as input

    #新增
    identifier='building_id', # the column name that identifies a building
    # identifier='District', # the column name that identifies a bs

    nan_constant=0, # the constant to transform nan values
    x_scaler='minmax', # x_scaler
    y_scaler='minmax', # y_scaler

    #新增
    targets=["meter_reading"],
    
    #新增
    outlier_detection=False, # whether to perform flooring and capping
    # outlier_detection=True, # whether to perform flooring and capping

    criterion='mse', # optimization criterion, mse or l1
    fl_rounds=600, # the number of federated rounds
    fraction=1., # the percentage of available client to consider for random selection


    aggregation="gate_aware_fedamp", # federated aggregation algorithm
    # aggregation="fedavg", # federated aggregation algorithm


    epochs=10, # the number of maximum local epochs
    lr=0.001, # learning rate
    optimizer='adam', # the optimizer, it can be sgd or adam
    batch_size=128, # the batch size to use
    local_early_stopping=False, # whether to use early stopping
    local_patience=50, # patience value for the early stopping parameter (if specified)
    max_grad_norm=0.0, # whether to clip grad norm
    reg1=0.0, # l1 regularization
    reg2=0.0, # l2 regularization
    # model_name='moe_lstm', # the model to use, it can be rnn, lstm, gru, cnn or da_encoder_decoder
    model_name='group_moe_lstm', # the model to use, it can be rnn, lstm, gru, cnn or da_encoder_decoder

    #新增
    group_feature_splits=None,

    cuda=True, # whether to use gpu
    
    seed=42, # reproducibility

    # here we define the exogenous data
    # assign_stats=["mean", "median", "std", "variance", "kurtosis", "skew"],
    assign_stats=[""],

    use_time_features=True, # whether to use datetime features
    # 新增对比学习开关
    use_contrastive=False,  # 是否启用对比学习


    #MOE模块参数
    # 对于 moe_lstm: num_experts=总专家数
    # 对于 group_moe_lstm: num_experts=每个特征组的专家数
    # Rigorous grouped-RL baseline (recommended for long expensive runs)
    # Keep experts/group moderate so each expert receives enough updates.
    num_experts=8,
    gating_hidden=64,
    gating_type='softmax',
    # 方案一：每组只看自己的特征（强制分工）
    group_use_full_features=False,
    group_diversity_enabled=True,
    group_diversity_lambda=0.02,

    # 特征组划分模式：
    # "two" 保持原始逻辑，基于“组内方差/组间方差”自动划分 dynamic/static；
    # "three" 继续使用该指标，按 ratio 从低到高数据驱动划分为 static/mixed/dynamic。
    group_split_mode="three",

    # 分组划分：基于“组内方差/组间方差”比值的阈值
    group_split_within_ratio=1.0,
    group_split_zero_ratio_eps=1e-8,
    # 两组重叠共享特征实验：仅在支持该逻辑的脚本中生效。
    # auto: 继续使用客户端统计汇总出的 within/between/ratio 自动挑选共享特征；
    # manual: 使用下面两个列表手工指定共享特征。
    group_overlap_enabled=True,
    group_overlap_mode="auto",
    group_overlap_scheme_name="two_group_auto_overlap_top2",
    group_overlap_dynamic_to_static=["meter_reading", "hour", "sin_hour", "cos_hour"],
    group_overlap_static_to_dynamic=["primary_use_code", "meter", "square_feet"],
    group_overlap_max_per_direction=2,
    group_overlap_max_dynamic_to_static=2,
    group_overlap_max_static_to_dynamic=2,

    # 推理期分组专家兜底：仅在灾难信号触发时，按本地验证集尝试候选组合
    inference_group_fallback_enabled=True,
    inference_group_fallback_r2_trigger=0.75,
    inference_group_fallback_nrmse_trigger=0.18,
    inference_group_fallback_max_candidates=128,
    inference_group_fallback_mse_trigger=0.004,
    inference_group_fallback_val_train_mse_ratio_trigger=1.45,
    inference_group_fallback_val_train_mse_min=0.002,
    inference_group_fallback_min_r2_gain=0.02,
    inference_group_fallback_min_mse_rel_gain=0.03,
    inference_group_fallback_selection_metric="r2",
    inference_group_fallback_min_raw_mse_rel_gain=0.0,
    inference_group_fallback_raw_mse_trigger=-1.0,


    # expert_threshold=0.15,  # gate 值阈值（用于 gate-aware 聚合）

    expert_threshold=0.25,  # gate 值阈值（用于 gate-aware 聚合），默认调低以更易选专家

    # 分组RL拥挤惩罚（相对负载）
    crowd_penalty_enabled=True,
    crowd_penalty_threshold_factor=1.45,  # 阈值 = 期望负载 * 系数
    crowd_penalty_scale=0.12,             # 每超出1个客户端的惩罚斜率
    crowd_penalty_min_load=2.0,           # 最小触发阈值
    crowd_penalty_max_abs=0.18,           # 每组惩罚最大绝对值（防止奖励被单项惩罚主导）
    crowd_penalty_warmup_rounds=12,

    # 分组动作负载感知重路由（仅在过载明显时触发，降低专家塌缩）
    load_aware_reroute_enabled=True,
    load_aware_reroute_start_round=40,
    load_aware_reroute_prob=0.65,
    load_aware_overload_factor=1.35,
    load_aware_underload_factor=0.85,
    load_aware_min_load=2.0,

    # 分组RL奖励权重（总和建议为1.0）
    reward_abs_weight=0.55,
    reward_delta_weight=0.30,
    reward_r2_weight=0.10,
    reward_mae_weight=0.05,
    # 分组奖励稳健化：前期仅使用 abs+delta，后期再启用质量项
    reward_group_warmup_rounds=60,
    reward_group_warmup_abs_weight=0.80,
    reward_group_warmup_delta_weight=0.20,
    reward_disable_r2_low_variance=True,
    reward_r2_min_target_std=0.08,

    # DQN选择器稳定性
    selector_eps_decay=400,
    selector_eps_end=0.05,
    # 400轮训练建议：约在200轮时衰减到 eps_end，后半程以利用为主。
    selector_group_eps_decay=200,
    selector_group_eps_end=0.08,
    # 分组专家初始化三阶段：bootstrap(结构化探索) -> 候选约束探索 -> 全空间探索
    selector_group_bootstrap_rounds=20,
    selector_group_candidate_until_round=200,
    selector_group_topk=3,
    selector_group_min_count=2,
    selector_updates_per_round_single=3,
    selector_updates_per_round_group=4,
    # 对比实验开关：默认开启 RL 选择器；关闭后可在 group 模式使用 soft/hard 固定路由
    selector_enabled=True,
    group_no_rl_route_mode="soft",  # soft | hard

    # Gate-Aware聚合：默认恢复为原始同专家普通平均。
    # True: 同一组同一专家的客户端直接 mean；无人选中时用本轮所有客户端 mean。
    # False: 启用下面的样本数/质量加权与低样本平滑实验逻辑。
    aggregation_legacy_expert_average=False,

    # Gate-Aware聚合：低样本专家平滑
    aggregation_low_sample_smoothing=True,
    aggregation_low_sample_min_clients=2, # 小于该客户端数认为低样本
    aggregation_low_sample_alpha=0.5,     # new = alpha*local + (1-alpha)*prev_global
    aggregation_low_sample_adaptive_alpha=False, # False: use fixed aggregation_low_sample_alpha
    aggregation_low_sample_lambda=2.0,
    # 专家使用度自适应平滑：alpha=n/(n+lambda)，选中客户端越少越保守。
    aggregation_usage_adaptive_smoothing=True,
    aggregation_usage_adaptive_lambda=2.0,

    # Gate-Aware聚合：Non-IID 可靠性加权。
    # 客户端仍全部参与聚合，但样本量更大、验证误差更低的更新权重更高。
    aggregation_sample_weight_enabled=False,
    aggregation_sample_weight_power=0.5,
    aggregation_quality_weight_enabled=False,
    aggregation_quality_metric="MSE",
    aggregation_quality_weight_power=1.0,
    aggregation_quality_eps=1e-8,
    aggregation_quality_min=0.25,
    aggregation_quality_max=4.0,

    # 服务器侧停滞早停（减少后期无效轮次）
    best_model_min_delta=1e-6,
    fl_early_stop_enabled=False,
    fl_early_stop_min_rounds=320,
    fl_early_stop_patience=180,



)


# 定义 get_model 函数
def get_model(model: str,
              input_dim: int,
              out_dim: int,
              lags: int = 10,
              exogenous_dim: int = 0,
              seed=42):
    if model == "rnn":
        model = RNN(input_dim=input_dim, rnn_hidden_size=128, num_rnn_layers=1, rnn_dropout=0.0,
                    layer_units=[128], num_outputs=out_dim, matrix_rep=True, exogenous_dim=exogenous_dim)
    elif model == "lstm":
        model = LSTM(input_dim=input_dim, lstm_hidden_size=128, num_lstm_layers=1, lstm_dropout=0.0,
                     layer_units=[128], num_outputs=out_dim, matrix_rep=True, exogenous_dim=exogenous_dim)
    elif model == "gru":
        model = GRU(input_dim=input_dim, gru_hidden_size=128, num_gru_layers=1, gru_dropout=0.0,
                    layer_units=[128], num_outputs=out_dim, matrix_rep=True, exogenous_dim=exogenous_dim)
    elif model == "cnn":
        model = CNN(num_features=input_dim, lags=lags, exogenous_dim=exogenous_dim, out_dim=out_dim)
    elif model == "da_encoder_decoder":
        model = DualAttentionAutoEncoder(input_dim=input_dim, architecture="lstm", matrix_rep=True)

    #新增MOE
    elif model == "moe_lstm":
            model = MoELSTM(input_dim=input_dim,
                            lstm_hidden_size=128,
                            num_lstm_layers=1,
                            lstm_dropout=0.0,
                            layer_units=[128],
                            num_outputs=out_dim,
                            matrix_rep=True,
                            exogenous_dim=exogenous_dim,
                            num_experts=args.num_experts,
                            gating_hidden=args.gating_hidden,
                            gating_type=args.gating_type)
            

    elif model == "group_moe_lstm":
        if not hasattr(args, "group_feature_splits") or not args.group_feature_splits:
            raise ValueError("group_moe_lstm 需要 args.group_feature_splits")
        model = GroupMoELSTM(
            input_dim=input_dim,
            lstm_hidden_size=128,
            num_lstm_layers=1,
            lstm_dropout=0.0,
            layer_units=[128],
            num_outputs=out_dim,
            exogenous_dim=exogenous_dim,
            group_feature_splits=args.group_feature_splits,
            gating_hidden=args.gating_hidden,
            experts_per_group=args.num_experts,
            use_full_features=bool(getattr(args, "group_use_full_features", True)),
            group_diversity_enabled=bool(getattr(args, "group_diversity_enabled", True))
        )


    else:
        raise NotImplementedError("Specified model is not implemented. Choose one from ['rnn', 'lstm', 'gru', 'cnn', 'da_encoder_decoder']")
    return model


# def get_input_dims(X_train, exogenous_data_train):
#     if args.model_name == "mlp":
#         input_dim = X_train.shape[1] * X_train.shape[2]
#     else:
#         input_dim = X_train.shape[2]
#
#     if exogenous_data_train is not None:
#         if len(exogenous_data_train) == 1:
#             cid = next(iter(exogenous_data_train.keys()))
#             exogenous_dim = exogenous_data_train[cid].shape[1]
#         else:
#             exogenous_dim = exogenous_data_train["all"].shape[1]
#     else:
#         exogenous_dim = 0
#
#     return input_dim, exogenous_dim
