"""
Implements the Client.
"""

import sys

from pathlib import Path
from ml.utils.model_utils import args

parent = Path(__file__).resolve().parents[2]
if parent not in sys.path:
    sys.path.insert(0, str(parent))

from logging import INFO, DEBUG
from typing import Dict, Tuple, List, Union, Optional, Any

from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.fl.client.client import Client
from ml.utils.logger import log
from ml.utils.train_utils import train, test
from ml.rl.dqn_selector import DQNExpertSelector
from ml.utils.helpers import get_criterion


class TorchRegressionClient(Client):
    def __init__(self, cid: Union[str, int], net: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 local_train_params: Optional[Dict[str, Union[str, int, float, bool]]]):
        self.cid = cid
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.initial_train_params = local_train_params
        self.epochs = None
        self.optimizer = None
        self.lr = None
        self.criterion = None
        self.early_stopping = None
        self.patience = None
        self.device = None
        self.reg1 = None
        self.reg2 = None
        self.max_grad_norm = None
        # self.fed_prox_mu = None
        self.fed_prox_mu = 0.01

        self.contrastive_lambda = 0.0005  # 新增，默认值
        self.selector = None  # 见下面初始化
        self.local_round_idx = 0
        self._cached_val_target_std = None
        self._init_local_train_params()

    def _init_local_train_params(self):

        self.epochs = self.initial_train_params["epochs"]
        self.optimizer = self.initial_train_params["optimizer"]
        self.lr = self.initial_train_params["lr"]

        crit = self.initial_train_params["criterion"]
        if isinstance(crit, str):
            try:
                self.criterion = get_criterion(crit.lower())
            except NotImplementedError:
                log(INFO, f"[{self.cid}] unknown criterion '{crit}', defaulting to MSELoss")
                self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = crit

        # self.criterion = self.initial_train_params["criterion"]





        self.early_stopping = self.initial_train_params["early_stopping"]
        self.patience = self.initial_train_params["patience"]
        self.device = self.initial_train_params["device"]
        try:
            self.reg1 = self.initial_train_params["reg1"]
        except KeyError:
            self.reg1 = 0.
        try:
            self.reg2 = self.initial_train_params["reg2"]
        except KeyError:
            self.reg2 = 0.
        try:
            self.max_grad_norm = self.initial_train_params["max_grad_norm"]
        except KeyError:
            self.max_grad_norm = 0.

        try:
            self.fed_prox_mu = self.initial_train_params["fedprox_mu"]
        except KeyError:
            self.fed_prox_mu = 0.




    def _collect_state(self) -> np.ndarray:
        """
        取训练集第一批样本最后一个 time‑step 的输入，
        concat exogenous（若有），对批次求均值得到状态向量。
        """
        # state_dim 由 selector 初始化时确定
        for xb, exb, y_hist, y in self.train_loader:
            xb = xb.to(self.device)
            exb = exb.to(self.device) if exb is not None else None

            x_last = xb[:, -1, :, 0]                    # (B, input_dim)
            if exb is not None:
                if exb.dim() == 3:
                    ex_last = exb[:, -1, :]
                else:
                    ex_last = exb
                gating_input = torch.cat([x_last, ex_last], dim=1)
            else:
                gating_input = x_last

            return gating_input.mean(dim=0).cpu().numpy()
        # 如果 loader 为空
        return np.zeros(getattr(self, "selector").net.fc[0].in_features if self.selector else 1,
                        dtype=np.float32)

    def _estimate_val_target_std(self, max_batches: int = 8) -> float:
        """Estimate target std on validation loader to detect low-variance clients."""
        if self._cached_val_target_std is not None:
            return float(self._cached_val_target_std)

        vals = []
        try:
            for bidx, batch in enumerate(self.val_loader):
                if bidx >= max_batches:
                    break
                if isinstance(batch, (list, tuple)) and len(batch) >= 4:
                    y = batch[3]
                else:
                    continue
                y_np = y.detach().cpu().numpy().reshape(-1)
                if y_np.size > 0:
                    vals.append(y_np)
        except Exception:
            vals = []

        if not vals:
            self._cached_val_target_std = 0.0
            return 0.0

        arr = np.concatenate(vals, axis=0)
        self._cached_val_target_std = float(np.std(arr))
        return float(self._cached_val_target_std)

    def _select_group_experts_from_gate(self) -> Optional[List[int]]:
        """No-RL hard route baseline: pick per-group expert by mean gate argmax on first train batch."""
        if not hasattr(self, "net"):
            return None
        num_groups = int(getattr(self.net, "num_groups", 0))
        experts_per_group = int(getattr(self.net, "experts_per_group", 0))
        if num_groups <= 0 or experts_per_group <= 0:
            return None

        try:
            self.net.eval()
            model_device = next(self.net.parameters()).device
            with torch.no_grad():
                for xb, exb, y_hist, yb in self.train_loader:
                    xb = xb.to(model_device)
                    exb = exb.to(model_device) if exb is not None else None
                    out = self.net(xb, exb, device=self.device, return_gates=True)
                    if not isinstance(out, tuple) or len(out) < 2:
                        return None
                    gates = out[1]  # [B, G*E]
                    if gates.dim() != 2:
                        return None
                    gates = gates.view(gates.size(0), num_groups, experts_per_group)
                    mean_gates = gates.mean(dim=0)  # [G, E]
                    selected = mean_gates.argmax(dim=1).detach().cpu().numpy().astype(np.int64).tolist()
                    return [int(e) for e in selected]
        except Exception as e:
            log(INFO, f"[{self.cid}] failed hard route from gate: {e}")
            return None
        finally:
            try:
                self.net.train()
            except Exception:
                pass
        return None

    def _ensure_group_action_stats(self, num_groups: int, experts_per_group: int):
        shape = (int(num_groups), int(experts_per_group))
        if not hasattr(self, "group_action_reward_sum") or getattr(self, "group_action_reward_sum", None) is None:
            self.group_action_reward_sum = np.zeros(shape, dtype=np.float32)
            self.group_action_count = np.zeros(shape, dtype=np.int32)
            return
        if self.group_action_reward_sum.shape != shape:
            self.group_action_reward_sum = np.zeros(shape, dtype=np.float32)
            self.group_action_count = np.zeros(shape, dtype=np.int32)

    def _update_group_action_stats(self, selected_experts: List[int], reward: float):
        if (not hasattr(self, "group_action_reward_sum")) or self.group_action_reward_sum is None:
            return
        for g, e in enumerate(selected_experts):
            g = int(g)
            e = int(e)
            if 0 <= g < self.group_action_reward_sum.shape[0] and 0 <= e < self.group_action_reward_sum.shape[1]:
                self.group_action_reward_sum[g, e] += float(reward)
                self.group_action_count[g, e] += 1

    def _build_group_candidates(self, num_groups: int, experts_per_group: int) -> Optional[List[List[int]]]:
        if (not hasattr(self, "group_action_reward_sum")) or self.group_action_reward_sum is None:
            return None

        topk = int(getattr(args, "selector_group_topk", 3))
        min_count = int(getattr(args, "selector_group_min_count", 2))
        topk = max(1, min(topk, experts_per_group))

        candidates = []
        for g in range(num_groups):
            cnt = self.group_action_count[g].astype(np.int32)
            rew = self.group_action_reward_sum[g].astype(np.float32)

            under_observed = [int(e) for e in range(experts_per_group) if int(cnt[e]) < min_count]
            under_observed = sorted(under_observed, key=lambda e: int(cnt[e]))

            avg = np.full((experts_per_group,), -1e9, dtype=np.float32)
            nonzero = cnt > 0
            avg[nonzero] = rew[nonzero] / np.maximum(cnt[nonzero], 1)
            ranked = np.argsort(-avg).astype(np.int64).tolist()

            cur = []
            for e in under_observed:
                if e not in cur:
                    cur.append(int(e))
                if len(cur) >= topk:
                    break
            for e in ranked:
                if len(cur) >= topk:
                    break
                if int(e) not in cur:
                    cur.append(int(e))

            if len(cur) == 0:
                cur = list(range(experts_per_group))
            candidates.append(cur)

        return candidates

    def _load_aware_reroute_group_actions(self,
                                          selected_experts: List[int],
                                          expert_load: Dict,
                                          state: Optional[np.ndarray]) -> Tuple[List[int], List[str]]:
        """Reroute overloaded group experts to underloaded ones with controlled probability."""
        if not bool(getattr(args, "load_aware_reroute_enabled", True)):
            return selected_experts, []
        if self.local_round_idx < int(getattr(args, "load_aware_reroute_start_round", 40)):
            return selected_experts, []
        if not isinstance(selected_experts, list) or len(selected_experts) == 0:
            return selected_experts, []
        if not expert_load:
            return selected_experts, []

        num_groups = int(getattr(self.net, "num_groups", len(selected_experts)))
        experts_per_group = int(getattr(self.net, "experts_per_group", 1))
        if experts_per_group <= 1:
            return selected_experts, []

        reroute_prob = float(getattr(args, "load_aware_reroute_prob", 0.65))
        overload_factor = float(getattr(args, "load_aware_overload_factor", 1.35))
        underload_factor = float(getattr(args, "load_aware_underload_factor", 0.85))
        min_threshold = float(getattr(args, "load_aware_min_load", 2.0))

        q_group = None
        try:
            if state is not None and self.selector is not None:
                s = torch.tensor(state, dtype=torch.float32, device=self.selector.device).unsqueeze(0)
                q_group = self.selector.net(s).view(1, num_groups, experts_per_group)[0].detach().cpu().numpy()
        except Exception:
            q_group = None

        def _load_of(g: int, e: int) -> float:
            key = f"g{g}_e{e}"
            if key in expert_load:
                return float(expert_load.get(key, 0.0))
            return float(expert_load.get(e, 0.0))

        rerouted = selected_experts.copy()
        reroute_logs = []

        for g in range(min(num_groups, len(rerouted))):
            loads = [float(_load_of(g, e)) for e in range(experts_per_group)]
            group_total = float(np.sum(loads))
            if group_total <= 0:
                continue

            expected = group_total / float(experts_per_group)
            overload_thr = max(min_threshold, expected * overload_factor)
            underload_thr = max(0.0, expected * underload_factor)

            cur_e = int(rerouted[g])
            cur_load = loads[cur_e]
            if cur_load <= overload_thr:
                continue
            if np.random.rand() > reroute_prob:
                continue

            candidates = [e for e in range(experts_per_group) if loads[e] <= underload_thr]
            if not candidates:
                continue

            min_load = min(loads[e] for e in candidates)
            min_load_candidates = [e for e in candidates if abs(loads[e] - min_load) < 1e-8]
            new_e = min_load_candidates[0]

            if q_group is not None and len(min_load_candidates) > 1:
                best_q = -1e18
                for e in min_load_candidates:
                    qv = float(q_group[g, e])
                    if qv > best_q:
                        best_q = qv
                        new_e = int(e)

            if new_e != cur_e:
                rerouted[g] = int(new_e)
                reroute_logs.append(
                    f"g{g}:e{cur_e}->{new_e} (load {cur_load:.2f}>{overload_thr:.2f}, target {loads[new_e]:.2f})"
                )

        return rerouted, reroute_logs



    def get_parameters(self) -> List[np.ndarray]:
    # def get_parameters(self) -> torch.nn.Module:
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_train_parameters(self,
                             params: Dict[str, Union[bool, str, int, float]],
                             verbose: bool = False):  # default parameters

        self.epochs = params["epochs"] if "epochs" in params else self.epochs
        self.optimizer = params["optimizer"] if "optimizer" in params else self.optimizer
        self.lr = params["lr"] if "lr" in params else self.lr

        if "criterion" in params:
            crit = params["criterion"]
            if isinstance(crit, str):
                try:
                    self.criterion = get_criterion(crit.lower())
                except NotImplementedError:
                    log(INFO, f"[{self.cid}] unknown criterion '{crit}', keeping old value")
            else:
                self.criterion = crit

        # self.criterion = params["criterion"] if "criterion" in params else self.criterion




        self.early_stopping = params["early_stopping"] if "early_stopping" in params else self.early_stopping
        self.patience = params["patience"] if "patience" in params else self.patience
        self.device = params["device"] if "device" in params else self.device
        self.reg1 = params["reg1"] if "reg1" in params else self.reg1
        self.reg2 = params["reg2"] if "reg2" in params else self.reg2
        self.max_grad_norm = params["max_grad_norm"] if "max_grad_norm" in params else self.max_grad_norm
        self.fed_prox_mu = params["fedprox_mu"] if "fedprox_mu" in params else self.fed_prox_mu

    # 新增：支持动态设置对比损失权重
        if "contrastive_lambda" in params:
            self.contrastive_lambda = params["contrastive_lambda"]




        if verbose:
            log(DEBUG, f"Training parameters change for client {self.cid}: "
                       f"epochs={self.epochs}, optimizer={self.optimizer}, lr={self.lr}, "
                       f"criterion={self.criterion}, early_stopping={self.early_stopping}, patience={self.patience}, "
                       f"device={self.device}, reg1={self.reg1}, reg2={self.reg2}, max_grad_norm={self.max_grad_norm}")

    # def set_parameters(self, parameters: Union[List[np.ndarray], torch.nn.Module]):
    #     if not isinstance(parameters, torch.nn.Module):
    #         params_dict = zip(self.net.state_dict().keys(), parameters)
    #         state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    #         self.net.load_state_dict(state_dict, strict=True)
    #     else:
    #         self.net.load_state_dict(parameters.state_dict(), strict=True)


    #新增
    # def set_parameters(self, parameters: Union[List[np.ndarray], torch.nn.Module]):
    #     if not isinstance(parameters, torch.nn.Module):
    #         local_state = self.net.state_dict()
    #         params_dict = zip(local_state.keys(), parameters)
    #         state_dict = OrderedDict()
    #         for k, v in params_dict:
    #             if k.startswith("gate."):
    #                 state_dict[k] = local_state[k]  # 保留本地 gate
    #             else:
    #                 state_dict[k] = torch.tensor(v).to(local_state[k].device)
    #         self.net.load_state_dict(state_dict, strict=True)
    #     else:
    #         # 合并 module：保留本地 gate，其他按 server module
    #         new_state = parameters.state_dict()
    #         local_state = self.net.state_dict()
    #         merged = OrderedDict()
    #         for k in local_state.keys():
    #             if k.startswith("gate."):
    #                 merged[k] = local_state[k]
    #             else:
    #                 merged[k] = new_state[k]
    #         self.net.load_state_dict(merged, strict=True)

    def set_parameters(self, parameters: Union[List[np.ndarray], torch.nn.Module]):
        if not isinstance(parameters, torch.nn.Module):
            local_state = self.net.state_dict()
            params_dict = zip(local_state.keys(), parameters)
            state_dict = OrderedDict()
            mismatches = 0
            for k, v in params_dict:
                if k.startswith("gate."):
                    state_dict[k] = local_state[k]  # 保留本地 gate
                else:
                    tensor = torch.tensor(v).to(local_state[k].device)
                    if tensor.shape == local_state[k].shape:
                        state_dict[k] = tensor
                    else:
                        mismatches += 1
            if mismatches:
                log(INFO, f"[{self.cid}] set_parameters: skip {mismatches} mismatched params (list)")
            self.net.load_state_dict(state_dict, strict=False)
        else:
            new_state = parameters.state_dict()
            local_state = self.net.state_dict()
            merged = OrderedDict()
            mismatches = 0
            for k in local_state.keys():
                if k.startswith("gate."):
                    merged[k] = local_state[k]
                else:
                    src = new_state.get(k, local_state[k])
                    if src.shape == local_state[k].shape:
                        merged[k] = src
                    else:
                        merged[k] = local_state[k]
                        mismatches += 1
            if mismatches:
                log(INFO, f"[{self.cid}] set_parameters: skip {mismatches} mismatched params (module)")
            self.net.load_state_dict(merged, strict=False)

    def _compute_mean_gates_on_traindata(self):
        import torch
        import numpy as np
        
        if not hasattr(self, 'net') or self.net is None:
            log(INFO, f"[{self.cid}] net is None, returning")
            # return None
            return None, 0

        log(INFO, f"[{self.cid}] has gate: {hasattr(self.net, 'gate')}, has experts: {hasattr(self.net, 'experts')}")
        
        self.net.eval()
        acc = None
        cnt = 0
        device = self.device
        batch_count = 0
        
        with torch.no_grad():
            for xb, exb, y_hist, yb in self.train_loader:
                batch_count += 1
                xb = xb.to(device)
                exb = exb.to(device) if exb is not None else None
                
                if hasattr(self.net, 'gate') and hasattr(self.net, 'experts'):
                    try:
                        # xb shape: (batch, seq_len, input_dim, 1)
                        # 取最后一个 timestep 的最后一个维度后 squeeze
                        x_last = xb[:, -1, :, 0]  # (batch, input_dim) — 2D！
                        
                        exog = None
                        if exb is not None:
                            if exb.dim() == 3:
                                exog = exb[:, -1, :]  # (batch, exog_dim)
                            else:
                                exog = exb  # 已经是 (batch, exog_dim)
                        
                        expected_exog_dim = self.net.exogenous_dim if hasattr(self.net, 'exogenous_dim') else 0
                        
                        if expected_exog_dim > 0:
                            if exog is None:
                                exog = torch.zeros(x_last.size(0), expected_exog_dim, device=device, dtype=x_last.dtype)
                            elif exog.size(1) < expected_exog_dim:
                                pad = torch.zeros(exog.size(0), expected_exog_dim - exog.size(1), device=device, dtype=exog.dtype)
                                exog = torch.cat([exog, pad], dim=1)
                            elif exog.size(1) > expected_exog_dim:
                                exog = exog[:, :expected_exog_dim]
                        
                        # 现在都是 2D 了，可以 concat
                        if expected_exog_dim > 0 and exog is not None:
                            gating_input = torch.cat([x_last, exog], dim=1)  # ✅ (batch, input_dim + exog_dim)
                        else:
                            gating_input = x_last  # ✅ (batch, input_dim)
                        
                        # gates = self.net.gate(gating_input)
                        # gates = torch.softmax(gates, dim=-1)
                        logits = self.net.gate(gating_input)
                        gate_temp = getattr(self.net, "gate_temp", 1.0)
                        gates = torch.softmax(logits / gate_temp, dim=-1)
                        acc = gates.sum(dim=0) if acc is None else acc + gates.sum(dim=0)
                        cnt += gates.size(0)
                        
                    except Exception as e:
                        log(INFO, f"[{self.cid}] batch {batch_count} ERROR: {type(e).__name__}: {e}")
                
                if cnt >= 512:
                    break
        
        if cnt > 0:
            mean_gates = (acc / cnt).cpu().numpy()
            log(INFO, f"[{self.cid}] computed mean_gates (n={cnt}): {mean_gates.tolist()}")
            # return mean_gates
            return mean_gates, int(cnt)
        else:
            log(INFO, f"[{self.cid}] no gates computed (cnt={cnt}, batch_count={batch_count})")
            # return None
            return None, 0

    def fit(self, model: Optional[Union[torch.nn.Module, List[np.ndarray]]] = None) -> Tuple[
        List[np.ndarray], int, float, Dict[str, float], int, float, Dict[str, float]]:
        if hasattr(self.net, "num_experts"):
            log(INFO, f"[{self.cid}] Model num_experts: {self.net.num_experts}")
        else:
            log(INFO, f"[{self.cid}] Model num_experts: N/A")
        self.local_round_idx += 1
        if model is not None:
            self.set_parameters(model)

        # 读取上轮专家负载（由 server.aggregate_models3 附上）
        _expert_load = getattr(model, 'expert_load', {}) if model is not None else {}       

        # ✅ 新增：确保 criterion 是函数而非字符串
        if isinstance(self.criterion, str):
            try:
                self.criterion = getattr(torch.nn, self.criterion)()
                log(INFO, f"[{self.cid}] converted criterion string to {type(self.criterion).__name__}")
            except Exception as e:
                log(INFO, f"[{self.cid}] failed to convert criterion: {e}, defaulting to MSELoss")
                self.criterion = torch.nn.MSELoss()






        # ==== 选专家：强化学习代替 max‑prob ====
        selected_expert = None
        selected_experts = None

        # 初始化 selector（第一次知道 num_experts 及状态维度）
        if self.selector is None and hasattr(self.net, "num_experts"):
            # gate_input_dim 在 MoELSTM 中有属性；否则使用 input+exog
            state_dim = getattr(self.net, "gate_input_dim",
                                getattr(self.net, "input_dim", 0)
                                + getattr(self.net, "exogenous_dim", 0))
            num_groups = int(getattr(self.net, "num_groups", 1))
            experts_per_group = int(getattr(self.net, "experts_per_group", getattr(self.net, "num_experts", 1)))
            # 调整超参数，参考上文 dqn_selector.py
            self.selector = DQNExpertSelector(state_dim,
                                              self.net.num_experts,
                                              gamma=0.5,
                                              eps_decay=int(getattr(args, "selector_group_eps_decay", 800) if num_groups > 1
                                                            else getattr(args, "selector_eps_decay", 400)),
                                              eps_end=float(getattr(args, "selector_group_eps_end", 0.10) if num_groups > 1
                                                            else getattr(args, "selector_eps_end", 0.05)),
                                              buffer_size=1000,
                                              batch_size=16,
                                              target_update=50,
                                              num_groups=num_groups,
                                              experts_per_group=experts_per_group)
            # self.selector = DQNExpertSelector(state_dim,
            #                                   self.net.num_experts,
            #                                   eps_decay=500)  # 可调整
        # # 计算 mean_gates 仍可做诊断/提交给 server
        # if hasattr(self.net, "gate") and hasattr(self.net, "experts"):
        #     mean_gates, cnt = self._compute_mean_gates_on_traindata()
        #     if mean_gates is not None and cnt > 0:
        #         self.mean_gates = mean_gates

        # # 用 selector 决策
        # state = self._collect_state()
        # if self.selector is not None:
        #     selected_expert = self.selector.select(state)
        #     self.selected_expert = selected_expert
        #     # log(INFO, f"[{self.cid}] RL selector chose expert {selected_expert}")
        #     # ← 在日志中加入 eps
        #     log(INFO, f"[{self.cid}] RL selector chose expert {selected_expert} "
        #               f"eps={self.selector.eps:.4f}")

        is_group_mode = getattr(self.net, "use_group_experts", False)

        selected_expert = None
        selected_experts = None
        expert_prior = None
        state = None  # 供后面 selector.store_transition 使用

        selector_enabled = bool(getattr(args, "selector_enabled", True))
        no_rl_route_mode = str(getattr(args, "group_no_rl_route_mode", "soft")).lower()

        if (not is_group_mode) and (self.selector is not None) and selector_enabled:
            # 非 group 模式：DQN 选单专家
            state = self._collect_state()
            selected_expert = self.selector.select(state)
            self.selected_expert = selected_expert
            log(INFO, f"[{self.cid}] RL selector chose expert {selected_expert} eps={self.selector.eps:.4f}")
        elif is_group_mode and (self.selector is not None) and selector_enabled:
            # group 模式：每组选 1 个专家
            state = self._collect_state()

            num_groups = int(getattr(self.net, "num_groups", 1))
            experts_per_group = int(getattr(self.net, "experts_per_group", getattr(self.net, "num_experts", 1)))
            self._ensure_group_action_stats(num_groups=num_groups, experts_per_group=experts_per_group)

            bootstrap_rounds = int(getattr(args, "selector_group_bootstrap_rounds", 20))
            candidate_until_round = int(getattr(args, "selector_group_candidate_until_round", 200))

            if self.local_round_idx <= bootstrap_rounds:
                # 前期结构化探索：可复现、覆盖均匀，避免纯随机初始化。
                if hasattr(self.selector, "advance_epsilon"):
                    self.selector.advance_epsilon()
                cid_hash = sum(ord(ch) for ch in str(self.cid))
                selected_experts = [
                    int((cid_hash + self.local_round_idx + g) % experts_per_group)
                    for g in range(num_groups)
                ]
                log(INFO, f"[{self.cid}] bootstrap group-actions={selected_experts} round={self.local_round_idx}/{bootstrap_rounds}")
            elif self.local_round_idx <= candidate_until_round:
                candidates = self._build_group_candidates(num_groups=num_groups, experts_per_group=experts_per_group)
                selected_experts = self.selector.select(state, group_candidates=candidates)
                log(INFO, f"[{self.cid}] constrained group-actions candidates={candidates}")
            else:
                selected_experts = self.selector.select(state)

            selected_experts = np.asarray(selected_experts, dtype=np.int64).tolist()

            if _expert_load:
                selected_experts, reroute_logs = self._load_aware_reroute_group_actions(
                    selected_experts=selected_experts,
                    expert_load=_expert_load,
                    state=state,
                )
                if reroute_logs:
                    log(INFO, f"[{self.cid}] load-aware reroute applied: {'; '.join(reroute_logs)}")

            self.selected_experts = selected_experts
            self.selected_expert = selected_experts  # 兼容 server 字段名
            log(INFO, f"[{self.cid}] RL selector chose experts per group {selected_experts} eps={self.selector.eps:.4f}")
        elif is_group_mode and (not selector_enabled):
            if no_rl_route_mode == "hard":
                selected_experts = self._select_group_experts_from_gate()
                if selected_experts is not None:
                    self.selected_experts = selected_experts
                    self.selected_expert = selected_experts
                    log(INFO, f"[{self.cid}] No-RL hard route uses selected_experts={selected_experts}")
                else:
                    self.selected_experts = None
                    self.selected_expert = None
                    log(INFO, f"[{self.cid}] No-RL hard route fallback to soft mixture")
            else:
                self.selected_experts = None
                self.selected_expert = None
                log(INFO, f"[{self.cid}] No-RL soft route uses mixture output")
        else:
            # group 模式：不选单专家
            self.selected_expert = None
            self.selected_experts = None

        
        # 在调用 train 之前基于 val mse 生成 expert_prior
        # try:
        #     num_experts = getattr(self.net, "num_experts", 1)
        #     val_mses = []
        #     for e in range(num_experts):
        #         val_mse, _, _, _, _, _ = test(self.net, self.val_loader, None, device=self.device, expert_idx=e)
        #         val_mses.append(float(val_mse) + 1e-8)


        # try:
        #     num_experts = getattr(self.net, "num_experts", 1)
        #     val_mses = []
        #     for e in range(num_experts):
        #         # 传入 criterion，返回 (loss,mse,rmse,mae,r2,nrmse)
        #         _, v_mse, _, _, _, _ = test(self.net, self.val_loader,
        #                                     self.criterion,
        #                                     device=self.device,
        #                                     expert_idx=e)
        #         val_mses.append(float(v_mse) + 1e-8)

        #     temp = 0.1
        #     scores = np.exp(-np.array(val_mses) / temp)
        #     expert_prior = (scores / scores.sum()).tolist()
        #     log(INFO, f"[{self.cid}] expert_prior from val_mse: {expert_prior}")
        # except Exception as ex:
        #     expert_prior = None
        #     log(INFO, f"[{self.cid}] failed to compute expert_prior: {ex}")

        # expert_prior 计算已废弃，保持 None
        expert_prior = None

        
        # # # 调用 train
        # self.net: torch.nn.Module = train(
        #     model=self.net, 
        #     train_loader=self.train_loader, 
        #     test_loader=self.val_loader,
        #     epochs=self.epochs, 
        #     optimizer=self.optimizer,
        #     lr=self.lr, 
        #     criterion=self.criterion,
        #     early_stopping=self.early_stopping, 
        #     patience=self.patience,
        #     reg1=self.reg1, 
        #     reg2=self.reg2, 
        #     max_grad_norm=self.max_grad_norm,
        #     fedprox_mu=self.fed_prox_mu,
        #     log_per=10,
        #     contrastive_lambda=self.contrastive_lambda,
        #     expert_idx=selected_expert,
        #     update_gate=False,
        #     gate_loss_weight=0.0,  # ← 改成 0.0（用 RL 替代 gate loss）
        #     expert_prior=expert_prior,  # ← 新增参数

        # )

        self.net = train(
            model=self.net,
            train_loader=self.train_loader,
            test_loader=self.val_loader,
            epochs=self.epochs,
            optimizer=self.optimizer,
            lr=self.lr,
            criterion=self.criterion,
            early_stopping=self.early_stopping,
            patience=self.patience,
            reg1=self.reg1,
            reg2=self.reg2,
            max_grad_norm=self.max_grad_norm,
            fedprox_mu=self.fed_prox_mu,
            device=self.device,
            log_per=10,
            contrastive_lambda=self.contrastive_lambda,
            expert_idx=None if is_group_mode else selected_expert,
            selected_experts=selected_experts if is_group_mode else None,
            update_gate=False if is_group_mode else False,
            gate_loss_weight=0.0,
            expert_prior=expert_prior
        )


        # # ========== 训练结束后用多指标组合奖励更新 selector ==========
        # try:
        #     # _, val_mse, val_rmse, val_mae, val_r2, val_nrmse = test(
        #     #     self.net, self.val_loader, None,
        #     #     device=self.device,
        #     #     expert_idx=selected_expert
        #     # )
            
        #     # 传入 criterion，与前面计算 prior 时保持一致，
        #     # 返回 (loss, mse, rmse, mae, r2, nrmse)。
        #     _, val_mse, val_rmse, val_mae, val_r2, val_nrmse = test(
        #         self.net, self.val_loader, self.criterion,
        #         device=self.device,
        #         expert_idx=selected_expert
        #     )      
        #     # 维护历史以进行归一化
        #     if not hasattr(self, 'mse_history'):
        #         self.mse_history = []
        #         self.r2_history = []
            
        #     self.mse_history.append(val_mse)
        #     self.r2_history.append(val_r2)
            
        #     # 归一化奖励（使用最近 10 轮的均值和方差）
        #     if len(self.mse_history) > 1:
        #         mse_mean = np.mean(self.mse_history[-10:])
        #         mse_std = np.std(self.mse_history[-10:]) + 1e-8
        #         r2_mean = np.mean(self.r2_history[-10:])
        #         r2_std = np.std(self.r2_history[-10:]) + 1e-8
                
        #         # Z-score 正则化
        #         mse_norm = (val_mse - mse_mean) / mse_std
        #         r2_norm = (val_r2 - r2_mean) / r2_std
        #         mae_norm = val_mae / (mse_mean ** 0.5 + 1e-8)  # 近似
        #     else:
        #         mse_norm = -val_mse
        #         r2_norm = val_r2
        #         mae_norm = -val_mae
            
        #     # 多指标组合奖励（权重可自行调整）
        #     reward = (0.4 * (-mse_norm) +      # MSE 越小越好（已反向）
        #              1.0 * r2_norm +            # R² 越大越好
        #              0.2 * (-mae_norm))         # MAE 越小越好（已反向）
            
        #     next_state = self._collect_state()
        #     if self.selector is not None and selected_expert is not None:
        #         self.selector.store_transition(state, selected_expert,
        #                                        reward, next_state, done=False)
        #         self.selector.update()
        #         log(INFO, f"[{self.cid}] expert={selected_expert} "
        #                   f"R={reward:.4f} "
        #                   f"(mse={val_mse:.2f}, r2={val_r2:.4f}, mae={val_mae:.2f})")
        # except Exception as e:
        #     log(INFO, f"[{self.cid}] failed to update selector: {e}")
        # # ========== 新增结束 ==========

        if (not is_group_mode) and (self.selector is not None) and (selected_expert is not None):
            try:
                _, val_mse, val_rmse, val_mae, val_r2, val_nrmse = test(
                    self.net, self.val_loader, self.criterion,
                    device=self.device, expert_idx=selected_expert
                )

                if not hasattr(self, "init_val_mse") or self.init_val_mse <= 0:
                    self.init_val_mse = float(val_mse)
                if not hasattr(self, "ema_val_mse") or self.ema_val_mse <= 0:
                    self.ema_val_mse = float(val_mse)

                prev_ema_val_mse = float(self.ema_val_mse)

                absolute_term = np.tanh(
                    (self.init_val_mse - float(val_mse)) / (self.init_val_mse + 1e-8)
                )
                delta_term = np.tanh(
                    (prev_ema_val_mse - float(val_mse)) / (prev_ema_val_mse + 1e-8)
                )

                reward = 0.7 * absolute_term + 0.3 * delta_term
                self.ema_val_mse = 0.9 * self.ema_val_mse + 0.1 * float(val_mse)

                if _expert_load:
                    load = _expert_load.get(selected_expert, 1)
                    if load > 4:
                        crowd_penalty = -0.20 * (load - 4)
                        reward = max(-1.0, reward + crowd_penalty)
                        log(INFO, f"[{self.cid}] crowd_penalty={crowd_penalty:.3f} "
                                f"(expert {selected_expert} overloaded: {load} clients)")

                next_state = self._collect_state()
                self.selector.store_transition(state, selected_expert, reward, next_state, done=True)
                update_steps = int(getattr(args, "selector_updates_per_round_single", 3))
                for _ in range(max(1, update_steps)):
                    self.selector.update()

                log(INFO, f"[{self.cid}] expert={selected_expert} "
                        f"R={reward:.4f} "
                        f"(abs={absolute_term:.4f}, delta={delta_term:.4f}, "
                        f"mse={val_mse:.2f}, r2={val_r2:.4f}, mae={val_mae:.2f})")
            except Exception as e:
                log(INFO, f"[{self.cid}] failed to update selector: {e}")
        elif is_group_mode and (self.selector is not None) and (selected_experts is not None):
            try:
                _, val_mse, val_rmse, val_mae, val_r2, val_nrmse = test(
                    self.net, self.val_loader, self.criterion,
                    device=self.device, selected_experts=selected_experts
                )

                if not hasattr(self, "init_val_mse") or self.init_val_mse <= 0:
                    self.init_val_mse = float(val_mse)
                if not hasattr(self, "ema_val_mse") or self.ema_val_mse <= 0:
                    self.ema_val_mse = float(val_mse)
                if not hasattr(self, "init_val_mae") or self.init_val_mae <= 0:
                    self.init_val_mae = float(val_mae)

                prev_ema_val_mse = float(self.ema_val_mse)
                absolute_term = np.tanh((self.init_val_mse - float(val_mse)) / (self.init_val_mse + 1e-8))
                delta_term = np.tanh((prev_ema_val_mse - float(val_mse)) / (prev_ema_val_mse + 1e-8))

                r2_term = float(np.clip(val_r2, -1.0, 1.0))
                mae_term = np.tanh((self.init_val_mae - float(val_mae)) / (self.init_val_mae + 1e-8))

                # Robust staged reward design:
                # 1) warmup rounds: only abs+delta to reduce noisy multi-term credit assignment;
                # 2) later rounds: optionally enable r2/mae with low-variance r2 suppression.
                warmup_rounds = int(getattr(args, "reward_group_warmup_rounds", 60))
                if self.local_round_idx <= warmup_rounds:
                    w_abs = float(getattr(args, "reward_group_warmup_abs_weight", 0.80))
                    w_delta = float(getattr(args, "reward_group_warmup_delta_weight", 0.20))
                    w_r2 = 0.0
                    w_mae = 0.0
                else:
                    w_abs = float(getattr(args, "reward_abs_weight", 0.55))
                    w_delta = float(getattr(args, "reward_delta_weight", 0.30))
                    w_r2 = float(getattr(args, "reward_r2_weight", 0.10))
                    w_mae = float(getattr(args, "reward_mae_weight", 0.05))

                    if bool(getattr(args, "reward_disable_r2_low_variance", True)):
                        y_std = self._estimate_val_target_std(max_batches=8)
                        std_thr = float(getattr(args, "reward_r2_min_target_std", 0.08))
                        if y_std < std_thr:
                            w_r2 = 0.0

                w_sum = w_abs + w_delta + w_r2 + w_mae
                if w_sum > 1e-12:
                    w_abs, w_delta, w_r2, w_mae = [w / w_sum for w in (w_abs, w_delta, w_r2, w_mae)]
                else:
                    w_abs, w_delta, w_r2, w_mae = 0.8, 0.2, 0.0, 0.0

                reward = (w_abs * absolute_term
                          + w_delta * delta_term
                          + w_r2 * r2_term
                          + w_mae * mae_term)
                self.ema_val_mse = 0.9 * self.ema_val_mse + 0.1 * float(val_mse)

                # Group mode crowd penalty: discourage overloaded experts per group.
                crowd_warmup_rounds = int(getattr(args, "crowd_penalty_warmup_rounds", 20))
                if (self.local_round_idx > crowd_warmup_rounds
                        and getattr(args, "crowd_penalty_enabled", True) and _expert_load):
                    experts_per_group = int(getattr(self.net, "experts_per_group", getattr(args, "num_experts", 8)))
                    threshold_factor = float(getattr(args, "crowd_penalty_threshold_factor", 1.8))
                    penalty_scale = float(getattr(args, "crowd_penalty_scale", 0.15))
                    min_threshold = float(getattr(args, "crowd_penalty_min_load", 2.0))
                    penalty_cap = float(getattr(args, "crowd_penalty_max_abs", 0.20))

                    # 统计每组总负载，用于计算期望负载(=group_total/experts_per_group)
                    group_totals = {}
                    for k, v in _expert_load.items():
                        if isinstance(k, str) and k.startswith("g") and "_e" in k:
                            try:
                                g_part = k.split("_")[0]  # g0
                                g_idx = int(g_part[1:])
                                group_totals[g_idx] = group_totals.get(g_idx, 0.0) + float(v)
                            except Exception:
                                continue

                    group_penalties = []
                    overloaded = []
                    for g, e in enumerate(selected_experts):
                        e_int = int(e)
                        key = f"g{int(g)}_e{e_int}"
                        load = float(_expert_load.get(key, 0))
                        # Backward compatibility for legacy flat-load dict.
                        if load <= 0:
                            load = float(_expert_load.get(e_int, 0))

                        expected = 0.0
                        if experts_per_group > 0:
                            expected = float(group_totals.get(int(g), 0.0)) / float(experts_per_group)
                        threshold = max(min_threshold, expected * threshold_factor)

                        if load > threshold:
                            p = -penalty_scale * (load - threshold)
                            p = max(-penalty_cap, min(0.0, p))
                            group_penalties.append(p)
                            overloaded.append(f"g{g}e{e_int}:{load:.2f}>{threshold:.2f}")

                    if group_penalties:
                        crowd_penalty = float(np.mean(group_penalties))
                        reward = reward + crowd_penalty
                        log(INFO, f"[{self.cid}] group_crowd_penalty={crowd_penalty:.3f} "
                                f"(overloaded={','.join(overloaded)})")

                reward = float(np.clip(reward, -1.0, 1.0))

                self._update_group_action_stats(selected_experts=selected_experts, reward=reward)

                next_state = self._collect_state()
                self.selector.store_transition(state, np.asarray(selected_experts, dtype=np.int64), reward, next_state, done=True)
                update_steps = int(getattr(args, "selector_updates_per_round_group", 1))
                for _ in range(max(1, update_steps)):
                    self.selector.update()

                log(INFO, f"[{self.cid}] experts={selected_experts} R={reward:.4f} "
                        f"(abs={absolute_term:.4f}, delta={delta_term:.4f}, r2={r2_term:.4f}, mae_t={mae_term:.4f}, "
                        f"w=({w_abs:.2f},{w_delta:.2f},{w_r2:.2f},{w_mae:.2f}), "
                        f"mse={val_mse:.2f}, r2_raw={val_r2:.4f}, mae={val_mae:.2f})")
            except Exception as e:
                log(INFO, f"[{self.cid}] failed to update group selector: {e}")




    # # 训练后重新计算 mean_gates 并用作返回（保证 server 聚合依据与训练后 gate 行为一致）
    #     try:
    #         mean_gates_after, cnt_after = self._compute_mean_gates_on_traindata()
    #         if mean_gates_after is not None and cnt_after > 0:
    #             mean_gates = mean_gates_after
    #             # 保存训练后重新计算的 mean_gates（覆盖）
    #             self.mean_gates = mean_gates
    #             log(INFO, f"[{self.cid}] mean_gates AFTER train (n={cnt_after}): {mean_gates.tolist()}")
    #                     # 诊断：mean entropy & max prob
    #             p = mean_gates
    #             mean_entropy = -float((p * np.log(p + 1e-12)).sum())
    #             max_prob = float(np.max(p))
    #             log(INFO, f"[{self.cid}] mean_entropy AFTER train: {mean_entropy:.6f}, max_prob: {max_prob:.3f}")            
                
    #             # 基于训练后 mean_gates 确定 selected_expert（deterministic argmax）
    #             try:
    #                 mean_gates_tensor = torch.from_numpy(mean_gates).float().to(self.device)
    #                 min_samples_for_selection = 8
    #                 if cnt_after >= min_samples_for_selection:
    #                     selected_expert_after = int(mean_gates_tensor.argmax().item())
    #                     self.selected_expert = selected_expert_after
    #                     log(INFO, f"[{self.cid}] selected_expert AFTER train set to {self.selected_expert} (mean_gates={mean_gates.tolist()})")
    #             except Exception as e:
    #                 log(INFO, f"[{self.cid}] failed to set selected_expert AFTER train: {e}")
        
    #     except Exception as e:
    #         log(INFO, f"[{self.cid}] failed to recompute mean_gates AFTER train: {e}")




        
        _, train_loss, train_metrics = self.evaluate(self.train_loader)
        num_test, test_loss, test_metrics = self.evaluate(self.val_loader)

        # 返回时使用同一个 mean_gates
        # return self.get_parameters(), len(
        #     self.train_loader.dataset), train_loss, train_metrics, num_test, test_loss, test_metrics, mean_gates
        # 确保 selected_expert 存在于 client 对象，并返回它（可能为 None）
        self.selected_expert = getattr(self, "selected_expert", None)
        # return self.get_parameters(), len(
        #     self.train_loader.dataset), train_loss, train_metrics, num_test, test_loss, test_metrics,self.selected_expert
        return self.get_parameters(), len(self.train_loader.dataset), train_loss, train_metrics, num_test, test_loss, test_metrics, self.selected_expert
        # self.net: torch.nn.Module = train(model=self.net, train_loader=self.train_loader, test_loader=self.val_loader,
        #                                   epochs=self.epochs, optimizer=self.optimizer,
        #                                   lr=self.lr, criterion=self.criterion,
        #                                   early_stopping=self.early_stopping, patience=self.patience,
        #                                   reg1=self.reg1, reg2=self.reg2, max_grad_norm=self.max_grad_norm,
        #                                   fedprox_mu=self.fed_prox_mu,
        #                                   log_per=10,
        #                                   contrastive_lambda=self.contrastive_lambda  # 新增
        #                                   )
        # _, train_loss, train_metrics = self.evaluate(self.train_loader)
        # num_test, test_loss, test_metrics = self.evaluate(self.val_loader)

        # return self.get_parameters(), len(
        #     self.train_loader.dataset), train_loss, train_metrics, num_test, test_loss, test_metrics

    def evaluate(self, data: Optional[DataLoader] = None,
                 model: Optional[Union[torch.nn.Module, List[np.ndarray]]] = None,
                 params: Dict[str, Any] = None, method: Optional[str] = None, verbose: bool = False) -> Tuple[
        int, float, Dict[str, float]]:

        if not params or "criterion" not in params:
            params = dict()
            params["criterion"] = torch.nn.MSELoss()

        if model:
            self.set_parameters(model)

        if data is None and method == "test":
            data = self.val_loader
        if data is None and method == "train":
            data = self.train_loader

        # loss, mse, rmse, mae, r2, nrmse = test(self.net, data, params["criterion"], device=self.device)

        eval_selected_experts = getattr(self, "selected_experts", None)
        eval_selected_expert = getattr(self, "selected_expert", None)

        # For test-time/global evaluation, use greedy routing to avoid epsilon-greedy noise
        # contaminating best-round and early-stop decisions.
        if method == "test" and self.selector is not None:
            try:
                state = self._collect_state()
                greedy_action = self.selector.select_greedy(state)
                if getattr(self.net, "use_group_experts", False):
                    eval_selected_experts = np.asarray(greedy_action, dtype=np.int64).tolist()
                    eval_selected_expert = None
                else:
                    eval_selected_experts = None
                    eval_selected_expert = int(greedy_action)
            except Exception as e:
                log(INFO, f"[{self.cid}] evaluate(test) greedy routing fallback to cached selection: {e}")

        if eval_selected_experts is not None:
            loss, mse, rmse, mae, r2, nrmse = test(
                self.net, data, params["criterion"],
                device=self.device,
                selected_experts=eval_selected_experts
            )
        else:
            loss, mse, rmse, mae, r2, nrmse = test(self.net, data, params["criterion"],
                                                  device=self.device,
                                                  expert_idx=eval_selected_expert)






        metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R^2": r2, "NRMSE": nrmse}

        if verbose:
            log(INFO, f"[Client {self.cid} Evaluation on {len(data.dataset)} samples] "
                      f"loss: {loss}, mse: {mse}, rmse: {rmse}, mae: {mae}, nrmse: {nrmse}")

        return len(data.dataset), loss, metrics
