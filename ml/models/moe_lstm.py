import torch
import torch.nn as nn
from typing import List, Optional
from logging import INFO
from ml.utils.logger import log




class MoELSTM(nn.Module):
    """
    Mixture of Experts LSTM.
    保持与现有 LSTM 相近的构造器签名，新增 num_experts 与门控相关参数。
    forward(x, exogenous_data=None, return_gates=False) -> preds 或 (preds, gates)
    """


    def __init__(self,
                input_dim: int,
                lstm_hidden_size: int = 128,
                num_lstm_layers: int = 1,
                lstm_dropout: float = 0.0,
                layer_units: Optional[List[int]] = None,
                num_outputs: int = 1,
                matrix_rep: bool = True,
                exogenous_dim: int = 0,
                num_experts: int = 8,
                gating_hidden: int = 64,
                gating_type: str = "softmax"):
        super().__init__()

        # 基本属性
        self.input_dim = int(input_dim)
        self.lstm_hidden_size = int(lstm_hidden_size)
        self.num_lstm_layers = int(num_lstm_layers)
        self.num_experts = int(num_experts)
        self.exogenous_dim = int(exogenous_dim)
        self.layer_units = list(layer_units) if layer_units else []
        self.gating_type = gating_type

        # 专家：独立的 LSTM 列表
        self.experts = nn.ModuleList([
            nn.LSTM(input_size=self.input_dim,
                    hidden_size=self.lstm_hidden_size,
                    num_layers=self.num_lstm_layers,
                    dropout=lstm_dropout if self.num_lstm_layers > 1 else 0.0,
                    batch_first=True)
            for _ in range(self.num_experts)
        ])

        # gate 输入维度 = input_dim + exogenous_dim
        gate_in_dim = self.input_dim + self.exogenous_dim
        if gating_hidden and gating_hidden > 0:
            self.gate = nn.Sequential(
                nn.Linear(gate_in_dim, gating_hidden),
                nn.ReLU(),
                nn.Linear(gating_hidden, self.num_experts)
            )
        else:
            self.gate = nn.Linear(gate_in_dim, self.num_experts)
        self.softmax = nn.Softmax(dim=1)

        # fc 输入维度 = hidden + exogenous_dim
        fc_in_dim = self.lstm_hidden_size + self.exogenous_dim
        self.fc_in_dim = fc_in_dim  # 暴露以便诊断

        if len(self.layer_units) > 0:
            layers = []
            prev = fc_in_dim
            for u in self.layer_units:
                layers.append(nn.Linear(prev, u))
                layers.append(nn.ReLU())
                prev = u
            self.fc_body = nn.Sequential(*layers)
            self.head = nn.Linear(prev, num_outputs)
        else:
            self.fc_body = None
            self.head = nn.Linear(fc_in_dim, num_outputs)

        # 构造时的一致性检查：若不匹配则抛出明确错误，避免训练时反复 runtime shape 错误
        try:
            # 检查 gate 第一层输入维
            g_first = self.gate if isinstance(self.gate, nn.Linear) else next(m for m in self.gate if isinstance(m, nn.Linear))
            assert g_first.in_features == gate_in_dim, f"gate first layer in_features={g_first.in_features} != expected {gate_in_dim}"
            # 检查 fc_body（若存在）
            if self.fc_body is not None:
                fc_first = next(m for m in self.fc_body if isinstance(m, nn.Linear))
                assert fc_first.in_features == fc_in_dim, f"fc_body first layer in_features={fc_first.in_features} != expected {fc_in_dim}"
        except AssertionError as e:
            raise RuntimeError(f"MoELSTM init failed consistency check: {e}")
        

        # ✅ 保存 gate_input_dim 供 RL selector 使用
        self.gate_input_dim = gate_in_dim




      # ✅ 诊断输出：在 __init__ 最后加这个
        log(INFO, f"[MoELSTM] Parameter structure:")
        log(INFO, f"  num_experts: {self.num_experts}")
        log(INFO, f"  gate_in_dim: {self.gate[0].in_features if isinstance(self.gate, nn.Sequential) else self.gate.in_features}")
        
        total_params = 0
        for name, param in self.named_parameters():
            total_params += param.numel()
            log(INFO, f"    {name}: {param.shape}")
        
        log(INFO, f"  Total parameters: {total_params}")
        
        state_dict = self.state_dict()
        log(INFO, f"  State dict keys: {len(state_dict)}")
        for i, key in enumerate(state_dict.keys()):
            log(INFO, f"    [{i}] {key}: {state_dict[key].shape}")       
    
    # def __init__(self,
    #              input_dim: int,
    #              lstm_hidden_size: int = 128,
    #              num_lstm_layers: int = 1,
    #              lstm_dropout: float = 0.0,
    #              layer_units: Optional[List[int]] = None,
    #              num_outputs: int = 1,
    #              matrix_rep: bool = True,
    #              exogenous_dim: int = 0,
    #              num_experts: int = 3,
    #              gating_hidden: int = 64,
    #              gating_type: str = "softmax"):
    #     super().__init__()
    #     self.input_dim = input_dim
    #     self.lstm_hidden_size = lstm_hidden_size
    #     self.num_lstm_layers = num_lstm_layers
    #     self.num_experts = num_experts
    #     self.exogenous_dim = exogenous_dim
    #     self.layer_units = layer_units or []
    #     self.gating_type = gating_type

    #     # 专家：每个专家为独立的 nn.LSTM
    #     self.experts = nn.ModuleList([
    #         nn.LSTM(input_size=input_dim,
    #                 hidden_size=lstm_hidden_size,
    #                 num_layers=num_lstm_layers,
    #                 dropout=lstm_dropout if num_lstm_layers > 1 else 0.0,
    #                 batch_first=True)
    #         for _ in range(num_experts)
    #     ])



    #     # gate 输入维：包含 input 和外生特征（若有）
    #     gate_in_dim = input_dim + (exogenous_dim if exogenous_dim else 0)
    #     if gating_hidden and gating_hidden > 0:
    #         self.gate = nn.Sequential(
    #             nn.Linear(gate_in_dim, gating_hidden),
    #             nn.ReLU(),
    #             nn.Linear(gating_hidden, num_experts)
    #         )
    #     else:
    #         self.gate = nn.Linear(gate_in_dim, num_experts)
    #     self.softmax = nn.Softmax(dim=1)

    #     # FC head：先把 expert 输出（hidden）与 exogenous 拼接后再接 FC 层
    #     fc_in_dim = lstm_hidden_size + (exogenous_dim if exogenous_dim else 0)
    #     if len(self.layer_units) > 0:
    #         layers = []
    #         prev = fc_in_dim
    #         for u in self.layer_units:
    #             layers.append(nn.Linear(prev, u))
    #             layers.append(nn.ReLU())
    #             prev = u
    #         self.fc_body = nn.Sequential(*layers)
    #         self.head = nn.Linear(prev, num_outputs)
    #     else:
    #         self.fc_body = None
    #         self.head = nn.Linear(fc_in_dim, num_outputs)




    # ...existing code...
    def forward(self,
                x: torch.Tensor,
                exogenous_data: Optional[torch.Tensor] = None,
                device=None,
                y_hist=None,
                return_gates: bool = False,
                expert_idx: Optional[int] = None,
                hard: bool = False,
                **kwargs):
        """
        前向传播（改进版）：
        - 输入会被移动到模型参数所在 device（避免 cpu/cuda mismatch）
        - exogenous 会被 pad/truncate 到 self.exogenous_dim
        - gating_input 会被 pad/truncate 到 gate 第一层期望的 in_features
        """
        # 兼容冗余 channel 维
        if x.dim() > 3:
            x = x.view(x.size(0), x.size(1), -1)

        # 确保输入与 model 参数在同一 device
        model_device = next(self.parameters()).device
        if device is not None:
            # prefer model_device for parameter ops; move inputs to model_device
            device = torch.device(device) if not isinstance(device, torch.device) else device
        x = x.to(model_device)
        if exogenous_data is not None:
            exogenous_data = exogenous_data.to(model_device)

        # 取最后 timestep 的基础输入
        x_last = x[:, -1, :]  # (batch, input_dim)

        # 处理 exogenous（若有）：取最后 timestep 或保持向量形式
        exog = None
        if exogenous_data is not None:
            if exogenous_data.dim() == 3:
                exog = exogenous_data[:, -1, :]
            else:
                exog = exogenous_data
            exog = exog.view(exog.size(0), -1)  # (batch, exog_dim_actual)

            # 将 exog 调整到模型期望的 exogenous_dim（pad 或截断）
            expected_exog = getattr(self, "exogenous_dim", None)
            if expected_exog is None:
                expected_exog = exog.size(1)
            if exog.size(1) < expected_exog:
                pad = torch.zeros(exog.size(0), expected_exog - exog.size(1), device=exog.device, dtype=exog.dtype)
                exog = torch.cat([exog, pad], dim=1)
                # import warnings
                # warnings.warn(f"exog dim {exog.size(1) - pad.size(1)} -> padded to expected {expected_exog}")
                log(INFO, f"MoELSTM: exog padded to expected {expected_exog}")


            elif exog.size(1) > expected_exog:
                # import warnings
                # warnings.warn(f"exog dim ({exog.size(1)}) > expected ({expected_exog}), truncating")
                log(INFO, f"MoELSTM: exog dim ({exog.size(1)}) > expected ({expected_exog}), truncating")
                exog = exog[:, :expected_exog]

        # 找到 gate 的第一个 Linear 层以确定期望输入维度
        gate_linear = None
        if isinstance(self.gate, nn.Linear):
            gate_linear = self.gate
        else:
            for m in self.gate:
                if isinstance(m, nn.Linear):
                    gate_linear = m
                    break
        gate_in = gate_linear.in_features if gate_linear is not None else None

        # 合并 x_last 与 exog（若有），再对齐到 gate_in：pad 或截断
        if exog is not None:
            combined = torch.cat([x_last, exog], dim=1)
        else:
            combined = x_last

        if gate_in is not None:
            cur = combined.size(1)
            if cur < gate_in:
                pad = torch.zeros(combined.size(0), gate_in - cur, device=combined.device, dtype=combined.dtype)
                combined = torch.cat([combined, pad], dim=1)
            elif cur > gate_in:
                # import warnings
                # warnings.warn(f"gating_input dim ({cur}) > gate.in_features ({gate_in}), truncating to match")

                log(INFO, f"MoELSTM: gating_input dim ({cur}) > gate.in_features ({gate_in}), truncating to match")



                combined = combined[:, :gate_in]

        gating_input = combined
        gate_logits = self.gate(gating_input)
        gates = self.softmax(gate_logits)  # (batch, num_experts)

        # 各专家前向：取每个 LSTM 最后一层的 hidden state h_n[-1]
        expert_reprs = []
        for expert in self.experts:
            # expert 参数已经在 model_device 上，x 已移动到 model_device
            out, (h_n, c_n) = expert(x)  # h_n: (num_layers, batch, hidden)
            expert_reprs.append(h_n[-1])  # (batch, hidden)

        # 对专家输出拼接 exogenous（使用调整后的 exog，确保维度等于 self.exogenous_dim）
        if exog is not None:
            # exog 已经 pad/truncate 到 expected_exog
            expert_reprs = [torch.cat([r, exog], dim=1) for r in expert_reprs]

        # 硬选择/软混合逻辑
        if expert_idx is not None:
            selected = int(expert_idx)
            if not (0 <= selected < self.num_experts):
                raise ValueError(f"expert_idx out of range: {selected}")
            fused = expert_reprs[selected]
        elif hard:
            selected = int(gates.mean(dim=0).argmax().item())
            fused = expert_reprs[selected]
        else:
            stacked = torch.stack(expert_reprs, dim=1)  # (batch, num_experts, hidden(+exog))
            gates_unsq = gates.unsqueeze(2)  # (batch, num_experts, 1)
            fused = (stacked * gates_unsq).sum(dim=1)  # (batch, hidden(+exog))

        # FC head
        if self.fc_body is not None:
            features = self.fc_body(fused)
        else:
            features = fused
        preds = self.head(features)  # (batch, num_outputs)

        if return_gates:
            return preds, gates
        return preds
    # ...existing code...