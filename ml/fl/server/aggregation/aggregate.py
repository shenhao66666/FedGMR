"""
Aggregation functions.
"""
import copy
import math
from typing import List, Tuple
from functools import reduce

import torch
import numpy as np
from ml.utils.model_utils import args

def _restore_gate_params(weights_list: List[np.ndarray], global_model: torch.nn.Module) -> List[np.ndarray]:
    """
    在按 global_model.state_dict().keys() 顺序的 weights_list 中，
    将所有以 'gate.' 开头的参数替换为 global_model 中相应参数的值（numpy 格式）。
    """
    gm_keys = list(global_model.state_dict().keys())
    for i, k in enumerate(gm_keys):
        if k.startswith("gate."):
            weights_list[i] = global_model.state_dict()[k].cpu().numpy()
    return weights_list




# def aggregate_gate_aware(
#     client_params_list,
#     client_mean_gates_list=None,
#     expert_threshold=None,
#     num_experts=None
# ):
def aggregate_gate_aware(
    client_params_list,
    client_mean_gates_list=None,
    expert_threshold=None,
    num_experts=None,
    param_keys=None,
    global_params=None,
    client_num_examples=None,
    client_quality_scores=None,
):
    """
    Gate-Aware FedAMP 聚合：根据客户端的 selected_expert 进行专家分配聚合。
    
    Args:
        client_params_list: 各客户端上传的参数列表 (numpy arrays)
        client_mean_gates_list: 各客户端的信息列表，现在是 {"selected_expert": int} 的字典
        expert_threshold: 废弃（RL方案下不需用）
        num_experts: 专家总数
        client_num_examples: 每个客户端训练样本量，用于样本量加权
        client_quality_scores: 每个客户端验证质量分数，越大表示本轮更新越可靠
    
    Returns:
        聚合后的全局参数列表
    """
    import torch
    import numpy as np
    from logging import INFO
    from ml.utils.logger import log
    
    if not client_params_list:
        return []
    
    n_clients = len(client_params_list)
    
    log(INFO, f"[aggregate_gate_aware] num_experts={num_experts}, total params={len(client_params_list[0])}")

    # ✅ 打印客户端选择（可为 int 或 list）
    log(INFO, "[Server-Aggregate] Client selected_expert:")
    for i, item in enumerate(client_mean_gates_list):
        if isinstance(item, dict):
            sel = item.get("selected_expert", None)
            log(INFO, f"  Client {i}: selected_expert={sel}")
        else:
            log(INFO, f"  Client {i}: item={item}")

    # assignments[i] 结构：{"flat": Optional[int], "groups": Optional[List[int]]}
    assignments = [{"flat": None, "groups": None} for _ in range(n_clients)]
    for i, item in enumerate(client_mean_gates_list):
        if item is None:
            continue
        if isinstance(item, dict):
            sel = item.get("selected_expert", None)
            if sel is not None:
                if isinstance(sel, np.ndarray):
                    sel = sel.astype(np.int64).tolist()
                if isinstance(sel, (list, tuple)):
                    assignments[i]["groups"] = [int(x) for x in sel]
                else:
                    assignments[i]["flat"] = int(sel)
        else:
            # 兼容旧的 numpy array 格式（如果有的话）
            if isinstance(item, np.ndarray):
                arr = np.asarray(item)
                argmax = int(arr.argmax())
                if expert_threshold is None or float(arr[argmax]) >= expert_threshold:
                    assignments[i]["flat"] = argmax

    # 从参数键推断 group 结构（适配 GroupMoELSTM 的 experts.g.e.*）
    num_groups_from_keys = None
    if param_keys is not None:
        gset = set()
        for k in param_keys:
            if isinstance(k, str) and k.startswith("experts."):
                parts = k.split(".")
                if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
                    gset.add(int(parts[1]))
        if len(gset) > 0:
            num_groups_from_keys = max(gset) + 1

    experts_per_group = None
    if num_groups_from_keys is not None and num_experts is not None and num_groups_from_keys > 0:
        experts_per_group = max(1, int(num_experts) // int(num_groups_from_keys))

    legacy_expert_average = bool(getattr(args, "aggregation_legacy_expert_average", True))
    sample_weight_enabled = (not legacy_expert_average) and bool(getattr(args, "aggregation_sample_weight_enabled", True))
    quality_weight_enabled = (not legacy_expert_average) and bool(getattr(args, "aggregation_quality_weight_enabled", True))
    weighted_aggregation_enabled = sample_weight_enabled or quality_weight_enabled
    sample_power = float(getattr(args, "aggregation_sample_weight_power", 0.5))
    quality_power = float(getattr(args, "aggregation_quality_weight_power", 1.0))

    sample_weights = np.ones(n_clients, dtype=np.float64)
    if sample_weight_enabled and client_num_examples is not None and len(client_num_examples) == n_clients:
        sample_weights = np.asarray(client_num_examples, dtype=np.float64)
        sample_weights = np.where(np.isfinite(sample_weights) & (sample_weights > 0), sample_weights, 1.0)
        sample_weights = np.power(sample_weights, sample_power)

    quality_weights = np.ones(n_clients, dtype=np.float64)
    if quality_weight_enabled and client_quality_scores is not None and len(client_quality_scores) == n_clients:
        quality_weights = np.asarray(client_quality_scores, dtype=np.float64)
        quality_weights = np.where(np.isfinite(quality_weights) & (quality_weights > 0), quality_weights, 1.0)
        quality_weights = np.power(quality_weights, quality_power)

    client_weights = sample_weights * quality_weights
    client_weights = np.where(np.isfinite(client_weights) & (client_weights > 0), client_weights, 1.0)
    client_weights = client_weights / (float(np.mean(client_weights)) + 1e-12)
    if legacy_expert_average:
        log(INFO, "[Aggregate] legacy expert average enabled: same selected expert -> plain mean; no active expert -> all-client plain mean")
    elif weighted_aggregation_enabled:
        log(INFO, f"[Aggregate] client aggregation weights: {[round(float(w), 4) for w in client_weights.tolist()]}")

    def _weighted_average(param_list, client_indices):
        if not client_indices:
            return None
        if legacy_expert_average or not weighted_aggregation_enabled:
            values = [param_list[i] for i in client_indices]
            return np.mean(values, axis=0)
        local_weights = client_weights[np.asarray(client_indices, dtype=np.int64)]
        if not np.isfinite(local_weights).all() or float(local_weights.sum()) <= 1e-12:
            local_weights = np.ones(len(client_indices), dtype=np.float64)
        values = [param_list[i] for i in client_indices]
        return np.average(values, axis=0, weights=local_weights)

    def _low_sample_alpha(active_clients):
        if bool(getattr(args, "aggregation_low_sample_adaptive_alpha", True)):
            lam = float(getattr(args, "aggregation_low_sample_lambda", 2.0))
            effective_n = float(len(active_clients))
            return effective_n / (effective_n + max(lam, 1e-8))
        return float(getattr(args, "aggregation_low_sample_alpha", 0.3))

    usage_adaptive_smoothing = (not legacy_expert_average) and bool(
        getattr(args, "aggregation_usage_adaptive_smoothing", False)
    )

    def _usage_adaptive_alpha(active_clients):
        lam = float(getattr(args, "aggregation_usage_adaptive_lambda", 2.0))
        effective_n = float(len(active_clients))
        return effective_n / (effective_n + max(lam, 1e-8))

    # # 聚合逻辑保持不变
    # aggregated_params = [None] * len(client_params_list[0])
    
    # # 参数结构解析
    # params_per_expert = 4
    # num_expert_params = num_experts * params_per_expert
    
    # for param_idx in range(len(client_params_list[0])):
    #     param_list = [client_params_list[i][param_idx] for i in range(n_clients)]
        
    #     is_expert_param = False
    #     is_gate_param = False
    #     expert_idx = -1
        
    #     # 识别参数类型
    #     if param_idx < num_expert_params:
    #         is_expert_param = True
    #         expert_idx = param_idx // params_per_expert
    #     else:
    #         is_gate_param = True
        
    #     if is_expert_param and 0 <= expert_idx < num_experts:
    #         # ✅ 修改点：使用 assignments 而不是 mean_gates 阈值判断
    #         active_clients = [i for i, a in enumerate(assignments) if a == expert_idx]
                     
    #         if len(active_clients) > 0:
    #             aggregated = np.mean([param_list[i] for i in active_clients], axis=0)
    #             aggregated_params[param_idx] = aggregated
    #             log(INFO, f"[Aggregate] Expert {expert_idx} (param {param_idx}): "
    #                       f"aggregated from {len(active_clients)} clients: {active_clients}")
    #         else:
    #             aggregated = np.mean(param_list, axis=0)
    #             aggregated_params[param_idx] = aggregated
    #             log(INFO, f"[Aggregate] Expert {expert_idx} (param {param_idx}): "
    #                       f"no active clients, using global average")
        
    #     elif is_gate_param:
    #         # Gate 参数不聚合，保持全局值（之后由 _restore_gate_params 处理）
    #         aggregated_params[param_idx] = param_list[0].copy()
    #         log(INFO, f"[Aggregate] Gate parameter (param {param_idx}): not aggregated (kept local)")
        
    #     else:
    #         aggregated = np.mean(param_list, axis=0)
    #         aggregated_params[param_idx] = aggregated
    #         log(INFO, f"[Aggregate] Other parameter (param {param_idx}): global average")
    aggregated_params = [None] * len(client_params_list[0])

    for param_idx in range(len(client_params_list[0])):
        param_list = [client_params_list[i][param_idx] for i in range(n_clients)]

        low_sample_smoothing = (not legacy_expert_average) and bool(getattr(args, "aggregation_low_sample_smoothing", True))
        low_sample_min_clients = int(getattr(args, "aggregation_low_sample_min_clients", 2))
        low_sample_alpha = float(getattr(args, "aggregation_low_sample_alpha", 0.3))

        key = param_keys[param_idx] if param_keys is not None else ""
        is_expert_param = False
        is_group_expert_param = False
        is_gate_param = False
        expert_idx = -1
        group_idx = -1
        expert_in_group = -1

        if key.startswith("experts."):
            is_expert_param = True
            parts = key.split(".")
            # GroupMoELSTM: experts.<group_idx>.<expert_idx>.*
            if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
                is_group_expert_param = True
                group_idx = int(parts[1])
                expert_in_group = int(parts[2])
            else:
                # 兼容旧 MoE: experts.<expert_idx>.*
                try:
                    expert_idx = int(parts[1])
                except Exception:
                    expert_idx = -1
        elif key.startswith("gate."):
            is_gate_param = True

        if is_group_expert_param:
            active_clients = []
            for i, a in enumerate(assignments):
                groups_sel = a.get("groups", None)
                if groups_sel is not None and group_idx < len(groups_sel):
                    if int(groups_sel[group_idx]) == expert_in_group:
                        active_clients.append(i)
                    continue

                # 兼容旧扁平索引：flat -> (g,e)
                flat_sel = a.get("flat", None)
                if flat_sel is not None and experts_per_group is not None:
                    g = int(flat_sel) // int(experts_per_group)
                    e = int(flat_sel) % int(experts_per_group)
                    if g == group_idx and e == expert_in_group:
                        active_clients.append(i)

            if len(active_clients) > 0:
                active_mean = _weighted_average(param_list, active_clients)
                use_smoothing = (
                    (usage_adaptive_smoothing or (
                        low_sample_smoothing
                        and (len(active_clients) < low_sample_min_clients)
                    ))
                    and (global_params is not None)
                )
                if use_smoothing:
                    global_param = global_params[param_idx]
                    if hasattr(global_param, "shape") and global_param.shape == active_mean.shape:
                        alpha = (
                            _usage_adaptive_alpha(active_clients)
                            if usage_adaptive_smoothing
                            else _low_sample_alpha(active_clients)
                        )
                        aggregated = alpha * active_mean + (1.0 - alpha) * global_param
                        smooth_name = "usage-adaptive smooth" if usage_adaptive_smoothing else "low-sample smooth"
                        log(INFO, f"[Aggregate] GroupExpert g{group_idx}e{expert_in_group} ({key}): "
                                  f"{smooth_name} ({len(active_clients)} clients, alpha={alpha:.2f})")
                    else:
                        aggregated = active_mean
                        log(INFO, f"[Aggregate] GroupExpert g{group_idx}e{expert_in_group} ({key}): "
                                  "skip expert smooth (shape mismatch)")
                else:
                    aggregated = active_mean
                aggregated_params[param_idx] = aggregated
                log(INFO, f"[Aggregate] GroupExpert g{group_idx}e{expert_in_group} ({key}): "
                          f"aggregated from {len(active_clients)} clients: {active_clients}")
            else:
                aggregated = _weighted_average(param_list, list(range(n_clients)))
                aggregated_params[param_idx] = aggregated
                log(INFO, f"[Aggregate] GroupExpert g{group_idx}e{expert_in_group} ({key}): "
                          f"no active clients, using all-client average")

        elif is_expert_param and 0 <= expert_idx < num_experts:
            active_clients = [i for i, a in enumerate(assignments) if a.get("flat", None) == expert_idx]

            if len(active_clients) > 0:
                active_mean = _weighted_average(param_list, active_clients)
                use_smoothing = (
                    (usage_adaptive_smoothing or (
                        low_sample_smoothing
                        and (len(active_clients) < low_sample_min_clients)
                    ))
                    and (global_params is not None)
                )
                if use_smoothing:
                    global_param = global_params[param_idx]
                    if hasattr(global_param, "shape") and global_param.shape == active_mean.shape:
                        alpha = (
                            _usage_adaptive_alpha(active_clients)
                            if usage_adaptive_smoothing
                            else _low_sample_alpha(active_clients)
                        )
                        aggregated = alpha * active_mean + (1.0 - alpha) * global_param
                        smooth_name = "usage-adaptive smooth" if usage_adaptive_smoothing else "low-sample smooth"
                        log(INFO, f"[Aggregate] Expert {expert_idx} ({key}): "
                                  f"{smooth_name} ({len(active_clients)} clients, alpha={alpha:.2f})")
                    else:
                        aggregated = active_mean
                        log(INFO, f"[Aggregate] Expert {expert_idx} ({key}): "
                                  "skip expert smooth (shape mismatch)")
                else:
                    aggregated = active_mean
                aggregated_params[param_idx] = aggregated
                log(INFO, f"[Aggregate] Expert {expert_idx} ({key}): "
                          f"aggregated from {len(active_clients)} clients: {active_clients}")
            else:
                aggregated = _weighted_average(param_list, list(range(n_clients)))
                aggregated_params[param_idx] = aggregated
                log(INFO, f"[Aggregate] Expert {expert_idx} ({key}): "
                          f"no active clients, using all-client average")

        elif is_gate_param:
            aggregated_params[param_idx] = param_list[0].copy()
            log(INFO, f"[Aggregate] Gate parameter ({key}): not aggregated (will restore global gate)")

        else:
            aggregated = _weighted_average(param_list, list(range(n_clients)))
            aggregated_params[param_idx] = aggregated
            if weighted_aggregation_enabled:
                log(INFO, f"[Aggregate] Shared parameter ({key}): weighted global average")
            else:
                log(INFO, f"[Aggregate] Shared parameter ({key}): global average")
    return aggregated_params



























# 新增模型聚合函数（DDPG）
def aggregate_with_ddpg(agent, client_features, client_models, global_model, previous_weights=None):
    """
    使用 DDPG 进行聚合。
    :param agent: DDPG 实例
    :param client_features: 各客户端状态特征列表
    :param client_models: 参与本轮聚合的客户端模型列表
    :param global_model: 全局模型
    :return: (global_w_dict, weights_array)
    """
    if not isinstance(global_model, torch.nn.Module):
        raise TypeError("global_model 应该是一个 torch.nn.Module 对象。")

    # 使用客户端特征计算动作/权重（示例以均值作为输入）
    states = np.array(client_features).mean(axis=0)
    weights = agent.get_action(states)
    weights = np.clip(weights, 0.01, None)
    if previous_weights is not None:
        weights = 0.9 * weights + 0.1 * previous_weights
    weights = weights / weights.sum()

    # 初始化累加字典
    global_w = {k: torch.zeros_like(v) for k, v in global_model.state_dict().items()}

    for (net, weight) in zip(client_models, weights):
        net_para = net.state_dict()
        for key in global_w:
            # 跳过 gate 参数的累加（确保 gate 保持 global 的原始值或后续保留）
            if key.startswith("gate."):
                continue
            global_w[key] += net_para[key] * weight

    # 恢复 gate 为 global_model 的原始值（不受客户端影响）
    for key in list(global_w.keys()):
        if key.startswith("gate."):
            global_w[key] = global_model.state_dict()[key].clone()

    return global_w, weights


def simple_aggregate(results: List[Tuple[List[np.ndarray], int]], global_model: torch.nn.Module) -> torch.nn.Module:
    """Compute a simple average and return a torch.nn.Module."""
    weights = [
        [layer for layer in weights] for weights, _ in results
    ]

    # 计算每一层的简单平均值
    weights_prime: List[np.ndarray] = [
        reduce(np.add, layer_updates) / len(weights)
        for layer_updates in zip(*weights)
    ]

    # 恢复 gate 参数为 global_model 中的原始值（不参与平均）
    weights_prime = _restore_gate_params(weights_prime, global_model)

    # 将 weights_prime 转换为 torch.nn.Module 并返回
    new_global_model = copy.deepcopy(global_model)
    new_global_model.load_state_dict({k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), weights_prime)})
    return new_global_model


def median_aggregate(results: List[Tuple[List[np.ndarray], int]], global_model: torch.nn.Module) -> torch.nn.Module:
    """Compute median across weights and return a torch.nn.Module."""
    weights = [
        [layer for layer in weights] for weights, _ in results
    ]

    weights_prime: List[np.ndarray] = [
        np.median(layer_updates, axis=0)
        for layer_updates in zip(*weights)
    ]

    # 恢复 gate 参数
    weights_prime = _restore_gate_params(weights_prime, global_model)

    new_global_model = copy.deepcopy(global_model)
    new_global_model.load_state_dict({k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), weights_prime)})
    return new_global_model


def fedavg_aggregate(results: List[Tuple[List[np.ndarray], int]], global_model: torch.nn.Module) -> torch.nn.Module:
    """Compute weighted average and return a torch.nn.Module."""
    num_examples_total = sum([num_examples for _, num_examples in results])

    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    weights_prime: List[np.ndarray] = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]

    # 恢复 gate 参数为 global_model 中的原始值（不参与平均）
    weights_prime = _restore_gate_params(weights_prime, global_model)

    new_global_model = copy.deepcopy(global_model)
    new_global_model.load_state_dict({k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), weights_prime)})
    return new_global_model


def fednova_aggregate(results: List[Tuple[List[np.ndarray], int]], global_model: torch.nn.Module, rho: float = 0.) -> torch.nn.Module:
    """Compute weighted average according to FedNova and return a torch.nn.Module."""
    num_examples = [num_examples for _, num_examples in results]
    num_examples_total = sum(num_examples)

    weights = [
        [layer for layer in weights] for weights, _ in results
    ]

    taus = copy.deepcopy(num_examples)
    alphas = [taus[i] - rho * (1 - math.pow(rho, taus[i])) / (1 - rho) / (1 - rho) for i in range(len(taus))]

    diffs = copy.deepcopy(weights)
    keys = list(global_model.state_dict().keys())
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            diffs[i][j] = (global_model.state_dict()[keys[j]].cpu().numpy() - weights[i][j]) / alphas[i]

    d_total_round = [np.zeros_like(global_model.state_dict()[key].cpu().numpy()) for key in keys]

    for i in range(len(diffs)):
        d_para = diffs[i]
        for j in range(len(diffs[i])):
            d_total_round[j] = np.add(d_total_round[j], d_para[j] * num_examples[i] / num_examples_total)

    coeff = 0.
    for i in range(len(diffs)):
        coeff = np.add(coeff, alphas[i] * num_examples[i] / num_examples_total)

    weights_prime: List[np.ndarray] = copy.deepcopy([v.cpu().numpy() for v in global_model.state_dict().values()])
    for i in range(len(weights_prime)):
        weights_prime[i] = np.subtract(weights_prime[i], coeff * d_total_round[i])

    # 恢复 gate 参数
    weights_prime = _restore_gate_params(weights_prime, global_model)

    new_global_model = copy.deepcopy(global_model)
    new_global_model.load_state_dict({k: torch.tensor(v) for k, v in zip(keys, weights_prime)})
    return new_global_model


def fedadagrad_aggregate(results: List[Tuple[List[np.ndarray], int]], global_model: torch.nn.Module,
                         m_t: List[np.ndarray] = None, v_t: List[np.ndarray] = None,
                         beta_1: float = 0., eta: float = 0.1, tau: float = 1e-2) -> Tuple[torch.nn.Module, List[np.ndarray], List[np.ndarray]]:
    """Computed weighted average according to FedAdagrad and return a torch.nn.Module."""
    fedavg_aggregated = fedavg_aggregate(results, global_model)

    delta_t: List[np.ndarray] = [
        x.cpu().numpy() - y.cpu().numpy() for x, y in zip(list(fedavg_aggregated.state_dict().values()), list(global_model.state_dict().values()))
    ]

    if not m_t:
        m_t = [np.zeros_like(x) for x in delta_t]
    m_t = [np.multiply(beta_1, x) + (1 - beta_1) * y for x, y in zip(m_t, delta_t)]

    if not v_t:
        v_t = [np.zeros_like(x) for x in delta_t]
    v_t = [x + np.multiply(y, y) for x, y in zip(v_t, delta_t)]

    new_weights = [
        x + eta * y / (np.sqrt(z) + tau)
        for x, y, z in zip([v.cpu().numpy() for v in global_model.state_dict().values()], m_t, v_t)
    ]

    # 恢复 gate 参数
    new_weights = _restore_gate_params(new_weights, global_model)

    new_global_model = copy.deepcopy(global_model)
    new_global_model.load_state_dict({k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), new_weights)})
    return new_global_model, m_t, v_t


def fedyogi_aggregate(results: List[Tuple[List[np.ndarray], int]], global_model: torch.nn.Module,
                      m_t: List[np.ndarray] = None, v_t: List[np.ndarray] = None,
                      beta_1: float = 0.9, beta_2: float = 0.99, eta: float = 0.01, tau: float = 1e-3) -> Tuple[torch.nn.Module, List[np.ndarray], List[np.ndarray]]:
    """Compute weighted average according to FedYogi and return a torch.nn.Module."""
    fedavg_aggregated = fedavg_aggregate(results, global_model)

    global_weights = [v.cpu().numpy() for v in global_model.state_dict().values()]
    delta_t: List[np.ndarray] = [
        np.nan_to_num(
            np.asarray(x.cpu().numpy(), dtype=np.float64) - np.asarray(y, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        for x, y in zip(list(fedavg_aggregated.state_dict().values()), global_weights)
    ]

    if not m_t:
        m_t = [np.zeros_like(x) for x in delta_t]
    m_t = [
        np.nan_to_num(
            beta_1 * np.asarray(x, dtype=np.float64) + (1 - beta_1) * np.asarray(y, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        for x, y in zip(m_t, delta_t)
    ]

    if not v_t:
        v_t = [np.zeros_like(x) for x in delta_t]
    # FedYogi updates the second moment with a sign correction instead of
    # monotonically subtracting squared deltas. This keeps v_t non-negative and
    # avoids NaNs in the adaptive denominator.
    v_t = [
        np.maximum(
            np.nan_to_num(
                np.asarray(x, dtype=np.float64)
                - (1 - beta_2)
                * np.square(np.asarray(y, dtype=np.float64))
                * np.sign(np.asarray(x, dtype=np.float64) - np.square(np.asarray(y, dtype=np.float64))),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            0.0,
        )
        for x, y in zip(v_t, delta_t)
    ]

    new_weights = [
        np.nan_to_num(
            np.asarray(x, dtype=np.float64) + eta * np.asarray(y, dtype=np.float64) / (np.sqrt(np.asarray(z, dtype=np.float64)) + tau),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.asarray(x).dtype, copy=False)
        for x, y, z in zip(global_weights, m_t, v_t)
    ]

    # 恢复 gate 参数
    new_weights = _restore_gate_params(new_weights, global_model)

    new_global_model = copy.deepcopy(global_model)
    new_global_model.load_state_dict({k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), new_weights)})
    return new_global_model, m_t, v_t


def fedadam_aggregate(results: List[Tuple[List[np.ndarray], int]], global_model: torch.nn.Module,
                      m_t: List[np.ndarray] = None, v_t: List[np.ndarray] = None,
                      beta_1: float = 0.9, beta_2: float = 0.99, eta: float = 0.01, tau: float = 1e-3) -> Tuple[torch.nn.Module, List[np.ndarray], List[np.ndarray]]:
    """Compute weighted average according to FedAdam and return a torch.nn.Module."""
    fedavg_aggregated = fedavg_aggregate(results, global_model)

    delta_t: List[np.ndarray] = [
        np.array(x.cpu().numpy()) - np.array(y.cpu().numpy()) for x, y in zip(list(fedavg_aggregated.state_dict().values()), list(global_model.state_dict().values()))
    ]

    if not m_t:
        m_t = [np.zeros_like(x) for x in delta_t]
    m_t = [np.multiply(beta_1, np.array(x)) + (1 - beta_1) * np.array(y) for x, y in zip(m_t, delta_t)]

    if not v_t:
        v_t = [np.zeros_like(x) for x in delta_t]
    v_t = [np.multiply(beta_2, np.array(x)) + (1 - beta_2) * np.multiply(np.array(y), np.array(y)) for x, y in zip(v_t, delta_t)]

    new_weights = [
        np.array(x) + eta * np.array(y) / (np.sqrt(np.array(z)) + tau)
        for x, y, z in zip([v.cpu().numpy() for v in global_model.state_dict().values()], m_t, v_t)
    ]

    # 恢复 gate 参数
    new_weights = _restore_gate_params(new_weights, global_model)

    new_global_model = copy.deepcopy(global_model)
    new_global_model.load_state_dict({k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), new_weights)})
    return new_global_model, m_t, v_t


def fedavgm_aggregate(results: List[Tuple[List[np.ndarray], int]], global_model: torch.nn.Module,
                      server_momentum: float = 0.9, momentum_vector: List[np.ndarray] = None, server_lr: float = 1.) -> Tuple[torch.nn.Module, List[np.ndarray]]:
    fedavg_aggregated = fedavg_aggregate(results, global_model)

    delta_t: List[np.ndarray] = [
        np.array(x.cpu().numpy()) - np.array(y.cpu().numpy()) for x, y in zip(list(fedavg_aggregated.state_dict().values()), list(global_model.state_dict().values()))
    ]

    if momentum_vector is None:
        momentum_vector = [np.zeros_like(x) for x in delta_t]
    momentum_vector = [server_momentum * np.array(x) + np.array(y) for x, y in zip(momentum_vector, delta_t)]

    new_weights = [
        np.array(x) - server_lr * np.array(y)
        for x, y in zip([v.cpu().numpy() for v in global_model.state_dict().values()], momentum_vector)
    ]

    # 恢复 gate 参数
    new_weights = _restore_gate_params(new_weights, global_model)

    new_global_model = copy.deepcopy(global_model)
    new_global_model.load_state_dict({k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), new_weights)})
    return new_global_model, momentum_vector
