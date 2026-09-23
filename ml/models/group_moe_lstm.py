import torch
import torch.nn as nn
from typing import Dict, List, Optional


class GroupMoELSTM(nn.Module):
    """
    分组专家版：
    - 每个专家看同一个输入张量，但只激活自己负责的特征索引（其余置零）
    - 并行得到各专家表示
    - 融合器（gate）输出组权重后做加权融合
    """
    def __init__(
        self,
        input_dim: int,
        lstm_hidden_size: int = 128,
        num_lstm_layers: int = 1,
        lstm_dropout: float = 0.0,
        layer_units: Optional[List[int]] = None,
        num_outputs: int = 1,
        exogenous_dim: int = 0,
        group_feature_splits: Optional[Dict[str, List[int]]] = None,
        gating_hidden: int = 64,
        experts_per_group: int = 8,
        use_full_features: bool = True,
        group_diversity_enabled: bool = True
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.exogenous_dim = int(exogenous_dim)
        self.hidden = int(lstm_hidden_size)
        self.num_layers = int(num_lstm_layers)
        self.layer_units = list(layer_units) if layer_units else []
        self.use_group_experts = True
        self.experts_per_group = int(experts_per_group)

        if not group_feature_splits:
            raise ValueError("group_feature_splits 不能为空")
        self.group_names = list(group_feature_splits.keys())
        self.group_feature_splits = group_feature_splits
        self.num_groups = len(self.group_names)
        # 为兼容现有日志/客户端，num_experts 暴露总专家数量
        self.num_experts = self.num_groups * self.experts_per_group
        self.total_experts = self.num_experts
        self.gate_input_dim = self.input_dim + self.exogenous_dim
        self.use_full_features = bool(use_full_features)
        self.group_diversity_enabled = bool(group_diversity_enabled)
        self.last_group_diversity_loss = None

        # 每组包含多个专家，每个专家仍看完整输入但只激活本组特征
        self.experts = nn.ModuleList()
        for _ in range(self.num_groups):
            group_experts = nn.ModuleList([
                nn.LSTM(
                    input_size=self.input_dim,
                    hidden_size=self.hidden,
                    num_layers=self.num_layers,
                    dropout=lstm_dropout if self.num_layers > 1 else 0.0,
                    batch_first=True
                ) for _ in range(self.experts_per_group)
            ])
            self.experts.append(group_experts)

        # 融合 gate：对每个组单独给出 experts_per_group 维打分
        self.gate = nn.Sequential(
            nn.Linear(self.gate_input_dim, gating_hidden),
            nn.ReLU(),
            nn.Linear(gating_hidden, self.total_experts)
        )

        # 每组输出一个 hidden，组间拼接后进入头部
        fc_in = self.hidden * self.num_groups + self.exogenous_dim
        if self.layer_units:
            layers = []
            prev = fc_in
            for u in self.layer_units:
                layers.append(nn.Linear(prev, u))
                layers.append(nn.ReLU())
                prev = u
            self.fc_body = nn.Sequential(*layers)
            self.head = nn.Linear(prev, num_outputs)
        else:
            self.fc_body = None
            self.head = nn.Linear(fc_in, num_outputs)

    def _mask_x(self, x: torch.Tensor, feat_idx: List[int]) -> torch.Tensor:
        # x: [B, T, F]
        masked = torch.zeros_like(x)
        if len(feat_idx) > 0:
            fdim = x.size(2)
            valid_idx = [int(i) for i in feat_idx if 0 <= int(i) < fdim]
            if len(valid_idx) > 0:
                masked[:, :, valid_idx] = x[:, :, valid_idx]
        return masked

    def forward(
        self,
        x: torch.Tensor,
        exogenous_data: Optional[torch.Tensor] = None,
        device=None,
        y_hist=None,
        return_gates: bool = False,
        expert_idx: Optional[int] = None,
        hard: bool = False,
        **kwargs
    ):
        if x.dim() > 3:
            x = x.view(x.size(0), x.size(1), -1)

        model_device = next(self.parameters()).device
        x = x.to(model_device)

        exog = None
        if exogenous_data is not None:
            exogenous_data = exogenous_data.to(model_device)
            exog = exogenous_data[:, -1, :] if exogenous_data.dim() == 3 else exogenous_data
            exog = exog.view(exog.size(0), -1)
            if exog.size(1) < self.exogenous_dim:
                pad = torch.zeros(exog.size(0), self.exogenous_dim - exog.size(1), device=model_device)
                exog = torch.cat([exog, pad], dim=1)
            elif exog.size(1) > self.exogenous_dim:
                exog = exog[:, :self.exogenous_dim]

        x_last = x[:, -1, :]
        gate_in = torch.cat([x_last, exog], dim=1) if exog is not None else x_last
        gate_logits = self.gate(gate_in).view(-1, self.num_groups, self.experts_per_group)
        # 组内归一化
        gates = torch.softmax(gate_logits, dim=2)  # [B, G, E]

        # 每组每个专家并行表示: group_reps[g] = [B, E, H]
        group_reps = []
        for g, gname in enumerate(self.group_names):
            if self.use_full_features:
                x_g = x
            else:
                idx = self.group_feature_splits[gname]
                x_g = self._mask_x(x, idx)
            reps_this_group = []
            for e in range(self.experts_per_group):
                _, (h_n, _) = self.experts[g][e](x_g)
                reps_this_group.append(h_n[-1])
            group_reps.append(torch.stack(reps_this_group, dim=1))

        selected_experts = kwargs.get("selected_experts", None)

        selected_group_outs = None
        if selected_experts is not None:
            if len(selected_experts) != self.num_groups:
                raise ValueError(f"selected_experts 长度应为 {self.num_groups}，当前为 {len(selected_experts)}")
            selected_group_outs = []
            for g in range(self.num_groups):
                e = int(selected_experts[g])
                if not (0 <= e < self.experts_per_group):
                    raise ValueError(f"group {g} 的 expert 索引越界: {e}")
                selected_group_outs.append(group_reps[g][:, e, :])
            fused = torch.cat(selected_group_outs, dim=1)
        elif expert_idx is not None:
            # 兼容旧接口：扁平索引 -> (group, expert_in_group)
            flat_idx = int(expert_idx)
            g = flat_idx // self.experts_per_group
            e = flat_idx % self.experts_per_group
            g = max(0, min(self.num_groups - 1, g))
            picked = group_reps[g][:, e, :]
            selected_group_outs = [picked for _ in range(self.num_groups)]
            fused = picked
            # 旧接口只选一组，为了形状一致，将其复制到所有组后拼接
            fused = torch.cat([fused for _ in range(self.num_groups)], dim=1)
        elif hard:
            # 每组选 gate 最大专家
            best_e = gates.argmax(dim=2)  # [B, G]
            selected_group_outs = []
            for g in range(self.num_groups):
                idx = best_e[:, g].unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden)
                picked = group_reps[g].gather(1, idx).squeeze(1)
                selected_group_outs.append(picked)
            fused = torch.cat(selected_group_outs, dim=1)
        else:
            # 软融合：每组内按 gate 加权，再组间拼接
            selected_group_outs = []
            for g in range(self.num_groups):
                probs = gates[:, g, :].unsqueeze(2)  # [B, E, 1]
                selected_group_outs.append((group_reps[g] * probs).sum(dim=1))
            fused = torch.cat(selected_group_outs, dim=1)

        # 组间表示差异正则：鼓励不同组学习互补表征，避免两组坍缩到同一表示。
        self.last_group_diversity_loss = None
        if self.group_diversity_enabled and selected_group_outs is not None and len(selected_group_outs) > 1:
            reps = torch.stack(selected_group_outs, dim=1)  # [B, G, H]
            reps = reps / (reps.norm(dim=2, keepdim=True) + 1e-8)
            sim = torch.bmm(reps, reps.transpose(1, 2))  # [B, G, G]
            gnum = sim.size(1)
            eye = torch.eye(gnum, device=sim.device, dtype=sim.dtype).unsqueeze(0)
            offdiag = (1.0 - eye)
            denom = max(1, gnum * (gnum - 1))
            self.last_group_diversity_loss = ((sim * offdiag) ** 2).sum() / (sim.size(0) * denom)

        if exog is not None:
            fused = torch.cat([fused, exog], dim=1)

        if self.fc_body is not None:
            fused = self.fc_body(fused)
        out = self.head(fused)

        if return_gates:
            return out, gates.reshape(gates.size(0), -1)
        return out

    def get_auxiliary_loss(self):
        return self.last_group_diversity_loss