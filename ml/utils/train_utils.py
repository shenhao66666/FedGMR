"""
Training pipeline.
"""

import sys
from pathlib import Path
import numpy as np

parent = Path(__file__).resolve().parents[2]
if parent not in sys.path:
    sys.path.insert(0, str(parent))

import copy
from logging import INFO
from typing import Tuple, List, Union, Optional

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from ml.utils.logger import log
from ml.utils.helpers import get_optim, get_criterion, EarlyStopping, accumulate_metric
from ml.utils.model_utils import args


def contrastive_loss(features, temperature=0.5):
    """对比学习损失"""
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    logits = similarity_matrix / temperature
    labels = torch.arange(features.size(0)).to(features.device)
    loss = F.cross_entropy(logits, labels)
    return loss


def train(model: torch.nn.Module,
          train_loader,
          test_loader,
          epochs: int = 10,
          optimizer: str = "adam",
          lr: float = 1e-3,
          reg1: float = 0.,
          reg2: float = 0.,
          max_grad_norm: float = 0.,
          criterion: str = "mse",
          early_stopping: bool = True,
          patience: int = 50,
          plot_history: bool = False,
          device="cuda",
          fedprox_mu: float = 0.,
          log_per: int = 1,
          use_carbontracker: bool = False,
          contrastive_lambda: float = 0.0005,
          expert_idx: Optional[int] = None,
          selected_experts: Optional[List[int]] = None,
          expert_prior: Optional[Union[list, np.ndarray]] = None,  # ← 新增参数
          update_gate: bool = True,
          gate_loss_weight: float = 1e-3,  # ← 新增：控制 gate 辅助损失权重
          gate_loss_warmup_epochs: int = 20,        # gate loss warmup
          gate_temp_start: float = 5.0,            # start temp (soft)
          gate_temp_end: float = 1.2,              # end temp (sharper)
          gate_temp_anneal_epochs: int = 50,       # anneal duration
          prior_alpha:  float = 0.0,                # one-hot vs prior interp
          entropy_coef: float = 0.005,              # smaller entropy penalty
          entropy_warmup_epochs: int = 5,
          entropy_explore: float = -0.02    





          ):
    """
    Trains a neural network defined as torch module.
    
    当 expert_idx 不为 None 时，硬选该专家进行前向，同时让 gate 学习为该专家分配高权重。
    """


    # ✅ 新增：处理 criterion 可能是对象或字符串的情况
    if isinstance(criterion, str):
        criterion_fn = get_criterion(criterion)
    elif isinstance(criterion, torch.nn.Module):
        criterion_fn = criterion
    else:
        log(INFO, f"Warning: unknown criterion type {type(criterion)}, defaulting to MSELoss")
        criterion_fn = torch.nn.MSELoss()


    best_model, best_loss, best_epoch = None, -1, -1
    train_loss_history, train_rmse_history = [], []
    test_loss_history, test_rmse_history = [], []
    
    if early_stopping:
        es_trace = True if log_per == 1 else False
        monitor = EarlyStopping(patience, trace=es_trace)
    
    cb_tracker = None
    if use_carbontracker:
        try:
            from carbontracker.tracker import CarbonTracker
            cb_tracker = CarbonTracker(epochs=epochs, components="all", verbose=1)
        except ImportError:
            pass

    # --- 冻结未选专家参数 ---
    frozen = []
    if expert_idx is not None and hasattr(model, "experts"):
        for i, exp in enumerate(model.experts):
            if i != expert_idx:
                for p in exp.parameters():
                    frozen.append((p, p.requires_grad))
                    p.requires_grad = False
    
    # --- 可选冻结 gate ---
    gate_saved = []
    if not update_gate and hasattr(model, "gate"):
        for p in model.gate.parameters():
            gate_saved.append((p, p.requires_grad))
            p.requires_grad = False

    # --- 构造待优化参数列表 ---
    if expert_idx is None:
        params_to_opt = [p for p in model.parameters() if p.requires_grad]
    else:
        params_to_opt = []
        params_to_opt += list(model.experts[expert_idx].parameters())
        if getattr(model, "fc_body", None) is not None:
            params_to_opt += list(model.fc_body.parameters())
        params_to_opt += list(model.head.parameters())
        if update_gate and hasattr(model, "gate"):
            params_to_opt += list(model.gate.parameters())
        params_to_opt = [p for p in params_to_opt if p.requires_grad]

    # --- 创建优化器 ---
    if isinstance(optimizer, str) and optimizer.lower() == "adam":
        optim = torch.optim.Adam(params_to_opt, lr=lr)
    elif isinstance(optimizer, str) and optimizer.lower() == "sgd":
        optim = torch.optim.SGD(params_to_opt, lr=lr)
    else:
        optim = get_optim(model, optimizer, lr)

    # criterion_fn = get_criterion(criterion)
    global_weight_collector = copy.deepcopy(list(model.parameters()))

    for epoch in range(epochs):
        if use_carbontracker and cb_tracker is not None:
            cb_tracker.epoch_start()
        
        model.to(device)
        model.train()
        epoch_loss = []
        
        # for x, exogenous, y_hist, y in train_loader:
        for batch_idx, (x, exogenous, y_hist, y) in enumerate(train_loader):

            x, y = x.to(device), y.to(device)
            y_hist = y_hist.to(device)
            if exogenous is not None and len(exogenous) > 0:
                exogenous = exogenous.to(device)
            else:
                exogenous = None
            
            optim.zero_grad()

             # ====== 核心改动：硬选 + Gate 学习 ======
            if expert_idx is not None and update_gate and hasattr(model, 'gate'):
                # 1) 硬选前向（用于主损失）
                y_pred_hard = model(x, exogenous, device, y_hist,
                                   expert_idx=expert_idx, hard=True)

                # 2) 软前向 / 计算 gate logits（用于 gate 辅助损失）
                y_pred_soft = model(x, exogenous, device, y_hist,
                                    expert_idx=None, hard=False)

                # 获取 gating input -> gate_logits
                gates = None
                gate_logits = None
                try:
                    x_last = x[:, -1, :, 0]
                    if exogenous is not None:
                        exog_last = exogenous[:, -1, :] if exogenous.dim() == 3 else exogenous
                        gating_input = torch.cat([x_last, exog_last], dim=1)
                    else:
                        gating_input = x_last

                    # 温度放软 + softmax
                    gate_logits = model.gate(gating_input)  # (B, num_experts)
                    # gate_temp = 1.0  # ↑ 增大温度让分布更平滑（可调）

                    # gate temp annealing: start soft -> end sharp
                    t = min(1.0, (epoch) / max(1, gate_temp_anneal_epochs))
                    gate_temp = gate_temp_start * (1 - t) + gate_temp_end * t





                    # 保存训练时的 gate 温度，供 client / inference 使用
                    try:
                        model.gate_temp = gate_temp
                    except Exception:
                        pass
                    gate_logits_temp = gate_logits / gate_temp
                    gates = torch.softmax(gate_logits_temp, dim=-1)
                except Exception as e:
                    gates = None
                    gate_logits = None
                    log(INFO, f"Warning: Failed to get gates: {e}")

                # 主损失（硬选）
                main_loss = criterion_fn(y_pred_hard, y)

                # Gate 辅助损失：label-smoothing + 熵正则
                gate_loss = 0.0
                if gates is not None and gate_logits is not None and gate_loss_weight > 0:
                    batch_size = gates.size(0)
                    num_experts = gates.size(1)

                    # # 更强的 label smoothing
                    # smooth_eps = 0.10
                    # off_value = smooth_eps / (num_experts - 1)
                    # on_value = 1.0 - smooth_eps
                    # target_gates = torch.full((batch_size, num_experts), off_value, device=device)
                    # target_gates[:, expert_idx] = on_value
                        # 优先使用 client 端传入的 expert_prior（soft target）
                    # if expert_prior is not None:
                    #     prior = torch.tensor(expert_prior, device=device, dtype=gates.dtype)
                    #     target_gates = prior.unsqueeze(0).repeat(batch_size, 1)
                    #     # 若同时有硬选 expert_idx，则以 alpha 插值 one-hot 与 prior
                    #     if expert_idx is not None:
                    #         onehot = torch.zeros((batch_size, num_experts), device=device, dtype=gates.dtype)
                    #         onehot[:, expert_idx] = 1.0
                    #         alpha = 0.7
                    #         target_gates = alpha * onehot + (1 - alpha) * target_gates
                    # gate loss warmup 权重
                    cur_gate_loss_weight = gate_loss_weight * min(1.0, (epoch + 1) / max(1, gate_loss_warmup_epochs))
                    # epoch 早期不使用 one-hot 强监督（只用 prior），晚期才插值 one-hot
                    cur_prior_alpha = prior_alpha if epoch >= gate_loss_warmup_epochs else 0.0
                    if expert_prior is not None:
                        prior = torch.tensor(expert_prior, device=device, dtype=gates.dtype)
                        target_gates = prior.unsqueeze(0).repeat(batch_size, 1)
                        if expert_idx is not None and cur_prior_alpha > 0.0:
                            onehot = torch.zeros((batch_size, num_experts), device=device, dtype=gates.dtype)
                            onehot[:, expert_idx] = 1.0
                            target_gates = cur_prior_alpha * onehot + (1 - cur_prior_alpha) * target_gates
                    else:
                        # 回退到 label smoothing
                        smooth_eps = 0.02
                        off_value = smooth_eps / (num_experts - 1)
                        on_value = 1.0 - smooth_eps
                        target_gates = torch.full((batch_size, num_experts), off_value, device=device)
                        target_gates[:, expert_idx] = on_value

                    # else:
                    #     # 较小的 label smoothing（默认回退）
                    #     smooth_eps = 0.02
                    #     off_value = smooth_eps / (num_experts - 1)
                    #     on_value = 1.0 - smooth_eps
                    #     target_gates = torch.full((batch_size, num_experts), off_value, device=device)
                    #     target_gates[:, expert_idx] = on_value



                    gate_loss = F.kl_div(
                        torch.log_softmax(gate_logits_temp, dim=-1),
                        target_gates,
                        reduction='batchmean'
                    )

                    # # 熵正则（鼓励更高熵，避免崩塌）
                    # entropy = -(gates * torch.log(gates + 1e-12)).sum(dim=1).mean()
                    # entropy_coef = 0.02  # 稍增大系数（可调 0.005-0.05）

                    # # 合并损失：注意 gate_loss_weight 仍应较小（在 client 端设置）
                    # main_loss = main_loss + gate_loss_weight * gate_loss - entropy_coef * entropy
                    # entropy = -(gates * torch.log(gates + 1e-12)).sum(dim=1).mean()
                    # entropy_coef = 0.005  # 缩小系数，仍鼓励高熵但不致使总 loss 变负
                    # main_loss = main_loss + gate_loss_weight * gate_loss - entropy_coef * entropy
                    # 纠正：惩罚高熵（鼓励更确定的路由）
                    # entropy = -(gates * torch.log(gates + 1e-12)).sum(dim=1).mean()
                    # entropy_coef = 0.02
                    # main_loss = main_loss + gate_loss_weight * gate_loss + entropy_coef * entropy

                    # # 诊断日志：周期性打印 gate_loss / entropy / avg_max_prob
                    # if batch_idx % 100 == 0:
                    #     avg_max_prob = float(gates.max(dim=1)[0].mean().detach().cpu().item())
                    #     log(INFO, f"Gate debug: epoch {epoch+1} batch {batch_idx}, gate_loss={gate_loss.item():.6e}, entropy={entropy.item():.6e}, avg_max_prob={avg_max_prob:.4f}")
                    # 采用 warmup 权重 cur_gate_loss_weight，并使用传入的 entropy_coef 参数
                    entropy = -(gates * torch.log(gates + 1e-12)).sum(dim=1).mean()
                    # 熵系数 schedule：前期使用负值鼓励探索
                    cur_entropy_coef = entropy_coef if epoch >= entropy_warmup_epochs else entropy_explore
                    main_loss = main_loss + cur_gate_loss_weight * gate_loss + cur_entropy_coef * entropy

                    # 诊断日志：显示当前 gate warmup、prior alpha、entropy_coef
                    if batch_idx % 100 == 0:
                        avg_max_prob = float(gates.max(dim=1)[0].mean().detach().cpu().item())
                        log(INFO, f"Gate debug: epoch {epoch+1} batch {batch_idx}, gate_loss={gate_loss.item():.6e}, entropy={entropy.item():.6e}, avg_max_prob={avg_max_prob:.4f}, cur_gate_w={cur_gate_loss_weight:.4e}, gate_temp={gate_temp:.3f}, cur_prior_alpha={cur_prior_alpha:.3f}, cur_entropy_coef={cur_entropy_coef:.4f}")



# ...existing code...
                loss = main_loss
            else:
                # 原逻辑：不硬选或不更新 gate
                if args.use_contrastive:
                    if selected_experts is not None:
                        out = model(x, exogenous, device, y_hist,
                                   return_features=True,
                                   selected_experts=selected_experts)
                    elif expert_idx is not None:
                        out = model(x, exogenous, device, y_hist,
                                   return_features=True,
                                   expert_idx=expert_idx,
                                   hard=True)
                    else:
                        out = model(x, exogenous, device, y_hist,
                                   return_features=True)
                    if isinstance(out, tuple):
                        y_pred, features = out
                    else:
                        y_pred = out
                        features = None
                else:
                    if selected_experts is not None:
                        y_pred = model(x, exogenous, device, y_hist,
                                      selected_experts=selected_experts)
                    elif expert_idx is not None:
                        y_pred = model(x, exogenous, device, y_hist,
                                      expert_idx=expert_idx,
                                      hard=True)
                    else:
                        y_pred = model(x, exogenous, device, y_hist)
                    features = None
                
                loss = criterion_fn(y_pred, y)
                
                # 对比损失
                if args.use_contrastive and features is not None:
                    loss += contrastive_lambda * contrastive_loss(features)

            # GroupMoE 组间差异正则：鼓励不同组学习互补表示。
            if getattr(args, "group_diversity_enabled", False):
                aux_loss_fn = getattr(model, "get_auxiliary_loss", None)
                if callable(aux_loss_fn):
                    aux_loss = aux_loss_fn()
                    if aux_loss is not None:
                        aux_lambda = float(getattr(args, "group_diversity_lambda", 0.0))
                        if aux_lambda > 0.0:
                            loss = loss + aux_lambda * aux_loss

            # === FedProx 正则 ===
            if fedprox_mu > 0.:
                fedprox_reg = 0.
                for param_index, param in enumerate(model.parameters()):
                    fedprox_reg += ((fedprox_mu / 2) * 
                                   torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += fedprox_reg
            
            # === L1/L2 正则 ===
            if reg1 > 0.:
                params = torch.cat([p.view(-1) for name, p in model.named_parameters() 
                                   if "bias" not in name])
                loss += reg1 * torch.norm(params, 1)
            if reg2 > 0.:
                params = torch.cat([p.view(-1) for name, p in model.named_parameters() 
                                   if "bias" not in name])
                loss += reg2 * torch.norm(params, 2)

            loss.backward()
            if max_grad_norm > 0.:
                torch.nn.utils.clip_grad_norm_(params_to_opt, max_grad_norm)
            optim.step()
            epoch_loss.append(loss.item())
        
        train_loss = sum(epoch_loss) / len(epoch_loss) if len(epoch_loss) > 0 else 0.0

        # === 评估 ===
        _, train_mse, train_rmse, train_mae, train_r2, train_nrmse = \
            test(model, train_loader, criterion_fn, device, expert_idx=expert_idx, selected_experts=selected_experts)
        test_loss, test_mse, test_rmse, test_mae, test_r2, test_nrmse = \
            test(model, test_loader, criterion_fn, device, expert_idx=expert_idx, selected_experts=selected_experts)

        if (epoch + 1) % log_per == 0:
            log(INFO, f"Epoch {epoch + 1} [Train]: loss {train_loss}, mse: {train_mse}, "
                      f"rmse: {train_rmse}, mae {train_mae}, r2: {train_r2}, nrmse: {train_nrmse}")
            log(INFO, f"Epoch {epoch + 1} [Test]: loss {test_loss}, mse: {test_mse}, "
                      f"rmse: {test_rmse}, mae {test_mae}, r2: {test_r2}, nrmse: {test_nrmse}")
        
        train_loss_history.append(train_mse)
        train_rmse_history.append(train_rmse)
        test_loss_history.append(test_mse)
        test_rmse_history.append(test_rmse)

        if early_stopping:
            monitor(test_loss, model)
            best_loss = abs(monitor.best_score)
            best_model = monitor.best_model
            if epoch + 1 > patience:
                best_epoch = epoch + 1
            elif epoch + 1 == epochs:
                best_epoch = epoch + 1 - monitor.counter
            else:
                best_epoch = epoch + 1 - patience
            if monitor.early_stop:
                log(INFO, "Early Stopping")
                break
        else:
            if best_loss == -1 or test_loss < best_loss:
                best_loss = test_loss
                best_model = copy.deepcopy(model)
                best_epoch = epoch + 1
        
        if use_carbontracker and cb_tracker is not None:
            cb_tracker.epoch_end()

    # 恢复 requires_grad 状态
    for p, prev in frozen:
        p.requires_grad = prev
    for p, prev in gate_saved:
        p.requires_grad = prev

    if plot_history:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_history, label="Train MSE")
        plt.plot(test_loss_history, label="Test MSE")
        plt.legend()
        plt.title("MSE Loss")
        plt.subplot(1, 2, 2)
        plt.plot(train_rmse_history, label="Train RMSE")
        plt.plot(test_rmse_history, label="Test RMSE")
        plt.legend()
        plt.title("RMSE")
        plt.show()
        plt.close()
    
    if early_stopping and epochs > patience:
        log(INFO, f"Best Loss: {best_loss}, Best epoch: {best_epoch}")
    else:
        log(INFO, f"Best Loss: {best_loss}")
    
    return best_model


def test(model, data, criterion, device="cuda", expert_idx: Optional[int] = None,
         selected_experts: Optional[List[int]] = None) -> Tuple:
    """
    Tests a trained model.
    
    Args:
        expert_idx: 若指定则评估时对整个批次硬选该专家
    """
    model.to(device)
    model.eval()
    y_true, y_pred = [], []
    loss = 0.
    
    with torch.no_grad():
        for x, exogenous, y_hist, y in data:
            x, y = x.to(device), y.to(device)
            y_hist = y_hist.to(device)
            if exogenous is not None and len(exogenous) > 0:
                exogenous = exogenous.to(device)
            else:
                exogenous = None
            
            if selected_experts is not None:
                out = model(x, exogenous, device=device, y_hist=y_hist,
                           selected_experts=selected_experts)
            elif expert_idx is not None:
                out = model(x, exogenous, device=device, y_hist=y_hist, 
                           expert_idx=expert_idx, hard=True)
            else:
                out = model(x, exogenous, device=device, y_hist=y_hist)
            
            if criterion is not None:
                loss += criterion(out, y).item()
            y_true.extend(y)
            y_pred.extend(out)

    loss /= len(data.dataset)

    y_true = torch.stack(y_true)
    y_pred = torch.stack(y_pred)
    mse, rmse, mae, r2, nrmse = accumulate_metric(y_true.cpu(), y_pred.cpu())
    
    if criterion is None:
        return mse, rmse, mae, r2, nrmse, y_pred

    return loss, mse, rmse, mae, r2, nrmse
