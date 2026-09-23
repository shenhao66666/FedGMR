"""
Implements the server and the federated process.
"""
import scipy.stats
import copy
import sys
import importlib

from pathlib import Path

import torch

from ml.fl.server.aggregation.aggregate import aggregate_with_ddpg

parent = Path(__file__).resolve().parents[3]
if parent not in sys.path:
    sys.path.insert(0, str(parent))

import time
from logging import DEBUG, INFO
from typing import Optional, Callable, List, Tuple, Dict, Union

import numpy as np
from torch.utils.data import DataLoader

from ml.fl.server.client_proxy import ClientProxy
from ml.fl.server.client_manager import ClientManager, SimpleClientManager

from ml.utils.logger import log
from ml.fl.history.history import History

from ml.fl.server.aggregation.aggregator import Aggregator
from ml.fl.defaults import weighted_loss_avg, weighted_metrics_avg




from ml.utils.model_utils import *

#暂时吧CNN放到这
class CNN(torch.nn.Module):
    def __init__(self,
                 num_features=11, lags=10, out_dim=1,
                 exogenous_dim: int = 0,
                 in_channels=[1, 16],
                 out_channels=[16, 32],
                 kernel_sizes=[(2, 3), (5, 3)],
                 pool_kernel_sizes=[(2, 1)]):
        super(CNN, self).__init__()
        assert len(in_channels) == len(out_channels) == len(kernel_sizes)
        self.activation = torch.nn.Tanh()
        self.num_lags = lags
        self.num_features = num_features
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0],
                                     kernel_size=kernel_sizes[0], padding="same")
        self.conv2 = torch.nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels[1],
                                     kernel_size=kernel_sizes[1], padding="same")
        self.pool = torch.nn.AvgPool2d(kernel_size=pool_kernel_sizes[0])
        kernel0, kernel1 = pool_kernel_sizes[-1][0], pool_kernel_sizes[-1][1]
        self.fc = torch.nn.Linear(
            in_features=(out_channels[1] * int(lags / kernel0) * int(num_features / kernel1)) + exogenous_dim,
            out_features=out_dim)

    def forward(self, x, exogenous_data=None, device=None, y_hist=None):
        if len(x.shape) > 2:
            x = x.view(x.size(0), x.size(3), x.size(1), x.size(2))
        else:
            x = x.view(x.size(0), 1, self.num_lags, self.num_features,)
        x = self.conv1(x)  # [batch_size]
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        # concatenate conv output with exogenous data
        if exogenous_data is not None and len(exogenous_data) > 0:
            x = torch.cat((x, exogenous_data), dim=1)

        x = self.fc(x)

        return x










class Server:
    def __init__(self,
                 client_proxies: List[ClientProxy],
                 X_train,  # 新增参数
                 exogenous_data_train,  # 新增参数
                 client_manager: Optional[ClientManager] = None,
                 aggregation: Optional[str] = None,
                 aggregation_params: Optional[Dict[str, Union[str, int, float, bool]]] = None,
                 weighted_loss_fn: Optional[Callable] = None,
                 weighted_metrics_fn: Optional[Callable] = None,
                 val_loader: Optional[DataLoader] = None,
                 local_params_fn: Optional[Callable] = None):


        self.X_train = X_train  # 保存传入的 X_train
        self.exogenous_data_train = exogenous_data_train  # 保存传入的 exogenous_data_train

        self.previous_weights = None  # 初始化 previous_weights

        # self.global_model = None
        self.global_model = self._initialize_global_model()
        self.best_model = None
        self.best_loss, self.best_epoch = np.inf, -1
        self.best_selected_experts_by_client = {}
        self.no_improve_rounds = 0



        self.aggregation = aggregation

        
        self.client_proxies = client_proxies
        self._initialize_client_manager(client_manager)  # initialize the client manager

        self.weighted_loss = weighted_loss_fn if weighted_loss_fn is not None else weighted_loss_avg
        self.weighted_metrics = weighted_metrics_fn if weighted_metrics_fn is not None else weighted_metrics_avg

        if aggregation is None:
            aggregation = "fedavg"
        self.aggregator = Aggregator(aggregation_alg=aggregation, params=aggregation_params)
        log(INFO, f"Aggregation algorithm: {repr(self.aggregator)}")

        self.val_loader = val_loader
        self.local_params_fn = local_params_fn

        if aggregation == "ddpg":
            try:
                ddpg_module = importlib.import_module("ml.fl.server.aggregation.DDPG")
                DDPG = getattr(ddpg_module, "DDPG")
            except Exception as e:
                raise ImportError("aggregation='ddpg' 需要 ml.fl.server.aggregation.DDPG.DDPG，但当前未找到该实现。") from e
            # 初始化 DDPG 实例
            self.ddpg_agent = DDPG(
                # state_dim=aggregation_params.get("state_dim", 10),  # 状态维度（如 mse、rmse 等）
                state_dim=10 if aggregation_params is None else aggregation_params.get("state_dim", 10),
                action_dim=len(client_proxies),  # 动作维度等于客户端数量
                device="cuda" if torch.cuda.is_available() else "cpu"  # 使用 GPU 或 CPU
            )
            self.reward_history = []  # 用于存储奖励历史



    def _initialize_global_model(self) -> torch.nn.Module:
    #     """初始化全局模型"""
    #     model_name = args.model_name
    #     input_dim = 11
    #     out_dim = 1
    #     lags = args.num_lags
    #     exogenous_dim = 4
    #     seed = args.seed
    #     # input_dim, exogenous_dim = get_input_dims(self.X_train, self.exogenous_data_train)  # 动态获取 input_dim 和 exogenous_dim
    #     model = get_model(model=model_name,
    #                     input_dim=input_dim,
    #                     out_dim=out_dim,
    #                     lags=lags,
    #                     exogenous_dim=exogenous_dim,
    #                     seed=seed)

    #     return model
        """初始化全局模型（自动推断 input_dim 和 exogenous_dim，并做一致性检查）"""
        model_name = args.model_name
        out_dim = 1
        lags = args.num_lags
        seed = args.seed

        # 推断 input_dim
        input_dim = getattr(args, "input_dim", 11)
        if getattr(self, "X_train", None) is not None:
            try:
                if isinstance(self.X_train, dict):
                    # X_train is a dict of client arrays
                    first_val = next(iter(self.X_train.values()))
                    if hasattr(first_val, "shape"):
                        if len(first_val.shape) >= 3:
                            input_dim = int(first_val.shape[2])
                        elif len(first_val.shape) >= 2:
                            input_dim = int(first_val.shape[1])
                else:
                    input_dim = int(self.X_train.shape[-1])
            except Exception:
                pass

        # 推断 exogenous_dim（默认为 0，当有外生数据时取其最后一个维度）
        exogenous_dim = 0
        if getattr(self, "exogenous_data_train", None) is not None:
            edt = self.exogenous_data_train
            try:
                if isinstance(edt, dict):
                    # 优先使用 "all" 键，其次用第一个非 None 的条目
                    if "all" in edt and edt["all"] is not None:
                        arr = np.asarray(edt["all"])
                    else:
                        arr = None
                        for v in edt.values():
                            if v is not None:
                                arr = np.asarray(v)
                                break
                    if arr is not None:
                        exogenous_dim = int(arr.shape[-1])
                else:
                    arr = np.asarray(edt)
                    exogenous_dim = int(arr.shape[-1])
            except Exception:
                exogenous_dim = 0

        log(INFO, f"Initializing global model: {model_name}, input_dim={input_dim}, exogenous_dim={exogenous_dim}, lags={lags}")

        model = get_model(model=model_name,
                          input_dim=input_dim,
                          out_dim=out_dim,
                          lags=lags,
                          exogenous_dim=exogenous_dim,
                          seed=seed)

        # 一致性检查：若模型有 gate，提醒不匹配情形
        try:
            gate_in = None
            if hasattr(model, "gate"):
                g = model.gate
                if isinstance(g, torch.nn.Linear):
                    gate_in = g.in_features
                else:
                    for m in g:
                        if isinstance(m, torch.nn.Linear):
                            gate_in = m.in_features
                            break
            if gate_in is not None and gate_in != (input_dim + exogenous_dim):
                log(INFO, f"WARNING: gate.in_features={gate_in} != input_dim+exogenous_dim={input_dim + exogenous_dim}")
        except Exception:
            pass

        return model
    




    def _initialize_client_manager(self, client_manager) -> None:
        """Initialize client manager""" 
        log(INFO, "Initializing client manager...")
        if client_manager is None:
            client_manager: ClientManager = SimpleClientManager()
            self.client_manager = client_manager
        else:
            self.client_manager = client_manager

        log(INFO, "Registering clients...")
        for client_proxy in self.client_proxies:  # register clients
            self.client_manager.register(client_proxy)

        log(INFO, "Client manager initialized!")

    def fit(self,
            num_rounds: int,
            fraction: float,
            fraction_args: Optional[Callable] = None,
            use_carbontracker: bool = False) -> Tuple[Dict[str, torch.Tensor], History]:
        """Run federated rounds for num_rounds rounds."""

        history = History()

        self.evaluate_round(fl_round=0, history=history)

        log(INFO, "Starting FL rounds")
        cb_tracker = None
        if use_carbontracker:
            try:
                from carbontracker.tracker import CarbonTracker
                cb_tracker = CarbonTracker(epochs=num_rounds, components="all", verbose=1)
            except ImportError:
                pass

        start_time = time.time()

        for fl_round in range(1, num_rounds + 1):
            if use_carbontracker and cb_tracker is not None:
                cb_tracker.epoch_start()
            # train and replace the previous global model
            self.fit_round(fl_round=fl_round,
                           fraction=fraction,
                           fraction_args=fraction_args,
                           history=history)
            if use_carbontracker and cb_tracker is not None:
                cb_tracker.epoch_end()
            # evaluate global model
            self.evaluate_round(fl_round=fl_round,
                                history=history)

            if bool(getattr(args, "fl_early_stop_enabled", True)):
                min_rounds = int(getattr(args, "fl_early_stop_min_rounds", 80))
                patience = int(getattr(args, "fl_early_stop_patience", 120))
                if fl_round >= min_rounds and self.no_improve_rounds >= patience:
                    log(INFO, f"[Early Stop] no validation improvement for {self.no_improve_rounds} rounds "
                              f"(patience={patience}), stop at round {fl_round}.")
                    break
        end_time = time.time()
        # log(INFO, history)
        log(INFO, f"Time passed: {end_time - start_time} seconds.")
        log(INFO, f"Best global model found on fl_round={self.best_epoch} with loss={self.best_loss}")

        # return self.best_model, history
        # return self.best_model.state_dict(), history
        return self.best_model, history

    def fit_round(self, fl_round: int,
                  fraction: float,
                  fraction_args: Optional[Callable],
                  history: History) -> None:
        """Perform a federated round, i.e.,
            1) Select a fraction of available clients.
            2) Instruct selected clients to execute local training.
            3) Receive updated parameters from clients and their corresponding evaluation
            4) Aggregate the local learned weights.
        """
        # Inform clients for local parameters change if any
        if self.local_params_fn:
            for client_proxy in self.client_proxies:
                client_proxy.set_train_parameters(self.local_params_fn(fl_round), verbose=True)

        # STEP 1: Select a fraction of available clients
        selected_clients = self.sample_clients(fl_round, fraction, fraction_args)

        # STEPS 2-3: Perform local training and receive updated parameters
        num_train_examples: List[int] = []
        num_test_examples: List[int] = []
        train_losses: Dict[str, float] = dict()
        test_losses: Dict[str, float] = dict()
        all_train_metrics: Dict[str, Dict[str, float]] = dict()
        all_test_metrics: Dict[str, Dict[str, float]] = dict()
        results: List[Tuple[List[np.ndarray], int]] = []

        # for client in selected_clients:
        #     res = self.fit_client(fl_round, client)
        #     model_params, num_train, train_loss, train_metrics, num_test, test_loss, test_metrics = res
        #     num_train_examples.append(num_train)
        #     num_test_examples.append(num_test)
        #     train_losses[client.cid] = train_loss
        #     test_losses[client.cid] = test_loss
        #     all_train_metrics[client.cid] = train_metrics
        #     all_test_metrics[client.cid] = test_metrics
        #     results.append((model_params, num_train))
        for client in selected_clients:
            res = self.fit_client(fl_round, client)
            # fit 返回 (params, num_train, train_loss, train_metrics,
            #           num_test, test_loss, test_metrics, selected_expert)
            # model_params, num_train, train_loss, train_metrics, num_test, test_loss, test_metrics, selected_expert = res
            model_params, num_train, train_loss, train_metrics, num_test, test_loss, test_metrics, selected_expert = res





            num_train_examples.append(num_train)
            num_test_examples.append(num_test)
            train_losses[client.cid] = train_loss
            test_losses[client.cid] = test_loss
            all_train_metrics[client.cid] = train_metrics
            all_test_metrics[client.cid] = test_metrics




            # 先写到 proxy 本身，再写到内部 client（若有）
            client.selected_expert = selected_expert
            if hasattr(client, "client"):
                try:
                    client.client.selected_expert = selected_expert
                except Exception:
                    pass






            # try:
            #     # SimpleClientProxy 有 client 属性
            #     if hasattr(client, "client"):
            #         client.client.selected_expert = selected_expert
            #     else:
            #         client.selected_expert = selected_expert
            # except Exception:
            #     client.selected_expert = selected_expert



            results.append((model_params, num_train))

        history.add_local_train_loss(train_losses, fl_round)
        history.add_local_train_metrics(all_train_metrics, fl_round)
        history.add_local_test_loss(test_losses, fl_round)
        history.add_local_test_metrics(all_test_metrics, fl_round)

        # # STEP 4: Aggregate local models
        # self.global_model = self.aggregate_models(fl_round, results)

        if self.aggregation == "ddpg":
            #新增强化学习聚合
            # STEP 4: Aggregate local models
            self.global_model = self.aggregate_models2(fl_round, results, history)
        elif self.aggregation == "gate_aware_fedamp":
            self.global_model = self.aggregate_models3(
                fl_round,
                results,
                selected_clients,
                all_test_metrics=all_test_metrics,
            )

            # self.global_model = self.aggregate_models3(fl_round, results)
        else:
            self.global_model = self.aggregate_models(fl_round, results)



        if self.best_model is None:
            self.best_model = copy.deepcopy(self.global_model)

    def sample_clients(self, fl_round: int, fraction: float,
                       fraction_args: Optional[Callable] = None) -> List[ClientProxy]:
        """Sample available clients."""
        if fraction_args is not None:
            fraction: float = fraction_args(fl_round)

        selected_clients: List[ClientProxy] = self.client_manager.sample(fraction)
        #log(DEBUG, f"[Global round {fl_round}] Sampled {len(selected_clients)} clients "
        #           f"(out of {self.client_manager.num_available(verbose=False)})")

        return selected_clients

    def fit_client(self,
                   fl_round: int,
                   client: ClientProxy) -> Tuple[
        List[np.ndarray], int, float, Dict[str, float], int, float, Dict[str, float]]:
        """Perform local training."""
        #log(INFO, f"[Global round {fl_round}] Fitting client {client.cid}")
        if fl_round == 1:
            fit_res = client.fit(None)
        else:
            fit_res = client.fit(model=self.global_model)

        return fit_res

    def aggregate_models(self, fl_round: int, results: List[Tuple[List[np.ndarray], int]]) -> torch.nn.Module:
        log(INFO, f"[Global round {fl_round}] Aggregating local models...")
        # aggregated_params = self.aggregator.aggregate(results, self.global_model)
        self.global_model = self.aggregator.aggregate(results, self.global_model)

        return self.global_model


    # def aggregate_models3(self, fl_round: int, results: List[Tuple[List[np.ndarray], int]]) -> torch.nn.Module:
    #     log(INFO, f"[Global round {fl_round}] Aggregating local models using Gate-Aware FedAMP...")
        
    #     from ml.fl.server.aggregation.aggregate import aggregate_gate_aware
        
    #     client_params_list = []
    #     client_mean_gates_list = []
        
    #     for model_params, num_train in results:
    #         client_params_list.append(model_params)
        
    #     # for client_proxy in self.client_proxies:
    #     #     mean_gates = client_proxy.mean_gates if hasattr(client_proxy, 'mean_gates') else None
    #     #     client_mean_gates_list.append(mean_gates)

    #     for client_proxy in self.client_proxies:
    #         if client_proxy in selected_clients:  # 只提取本轮被选中的客户端的 mean_gates
    #             mean_gates = client_proxy.mean_gates if hasattr(client_proxy, 'mean_gates') else None
    #         else:
    #             mean_gates = None
    #         client_mean_gates_list.append(mean_gates)


        
    #     num_experts = getattr(self.global_model, 'num_experts', 3)
    #     expert_threshold = getattr(args, 'expert_threshold', 0.15)
        
    #     aggregated_params = aggregate_gate_aware(
    #         client_params_list,
    #         client_mean_gates_list,
    #         expert_threshold=expert_threshold,
    #         num_experts=num_experts
    #     )
        
    #     self.global_model.load_state_dict({k: torch.tensor(v) if isinstance(v, np.ndarray) else v 
    #                                       for k, v in zip(self.global_model.state_dict().keys(), aggregated_params)})
        
    #     return self.global_model
    def aggregate_models3(
        self,
        fl_round: int,
        results: List[Tuple[List[np.ndarray], int]],
        selected_clients: List[ClientProxy],
        all_test_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> torch.nn.Module:
  
        log(INFO, f"[Global round {fl_round}] Aggregating local models using Gate-Aware FedAMP...")
        
        from ml.fl.server.aggregation.aggregate import aggregate_gate_aware, _restore_gate_params

        client_params_list = [params for params, _ in results]
        client_mean_gates_list = []
        for client_proxy in selected_clients:
            sel = getattr(client_proxy, "selected_expert", None)
            if sel is None and hasattr(client_proxy, "client"):
                sel = getattr(client_proxy.client, "selected_expert", None)
            # 只有 RL 情况下 sel 有意义，mean_gates 无需传
            client_mean_gates_list.append({"selected_expert": sel})

        num_experts = getattr(self.global_model, 'num_experts', args.num_experts)
        expert_threshold = args.expert_threshold

        client_num_examples = [num_train for _, num_train in results]

        metric_values = []
        quality_weight_enabled = (
            bool(getattr(args, "aggregation_quality_weight_enabled", True))
            and not bool(getattr(args, "aggregation_legacy_expert_average", True))
        )
        if quality_weight_enabled and all_test_metrics is not None:
            quality_metric = str(getattr(args, "aggregation_quality_metric", "MSE"))
            quality_eps = float(getattr(args, "aggregation_quality_eps", 1e-8))
            quality_min = float(getattr(args, "aggregation_quality_min", 0.25))
            quality_max = float(getattr(args, "aggregation_quality_max", 4.0))
            for client_proxy in selected_clients:
                metrics = all_test_metrics.get(client_proxy.cid, {})
                value = metrics.get(quality_metric, metrics.get(quality_metric.upper(), None))
                try:
                    value = float(value)
                except Exception:
                    value = None
                if value is not None and np.isfinite(value) and value >= 0.0:
                    metric_values.append(value)
                else:
                    metric_values.append(np.nan)

        client_quality_scores = None
        if metric_values:
            arr = np.asarray(metric_values, dtype=np.float64)
            finite = arr[np.isfinite(arr) & (arr >= 0.0)]
            if finite.size > 0:
                center = float(np.median(finite))
                scores = []
                for value in arr:
                    if np.isfinite(value) and value >= 0.0:
                        score = (center + quality_eps) / (float(value) + quality_eps)
                    else:
                        score = 1.0
                    score = max(quality_min, min(quality_max, score))
                    scores.append(score)
                client_quality_scores = scores
                log(INFO, f"[Aggregate] quality metric={quality_metric}, values={metric_values}, "
                          f"scores={[round(float(s), 4) for s in scores]}")
        
        # aggregated_params = aggregate_gate_aware(
        #     client_params_list,
        #     client_mean_gates_list,
        #     expert_threshold=expert_threshold,
        #     num_experts=num_experts
        # )

        aggregated_params = aggregate_gate_aware(
            client_params_list,
            client_mean_gates_list,
            expert_threshold=expert_threshold,
            num_experts=num_experts,
            param_keys=list(self.global_model.state_dict().keys()),
            global_params=[v.detach().cpu().numpy() for v in self.global_model.state_dict().values()],
            client_num_examples=client_num_examples,
            client_quality_scores=client_quality_scores,
        )

        # 恢复 gate 为 global_model 中的值，避免把任意客户端的 gate 覆盖到全局
        aggregated_params = _restore_gate_params(aggregated_params, self.global_model)
        
        model_state = self.global_model.state_dict()
        load_dict = {}
        mismatches = 0
        for key, value in zip(model_state.keys(), aggregated_params):
            tensor = torch.tensor(value) if isinstance(value, np.ndarray) else value
            if tensor.shape == model_state[key].shape:
                load_dict[key] = tensor
            else:
                mismatches += 1
        if mismatches:
            log(INFO, f"[Aggregate] skip {mismatches} mismatched params when loading global model")
        if load_dict:
            self.global_model.load_state_dict(load_dict, strict=False)
        

        # 统计下一轮拥塞惩罚使用的专家负载。
        # 优先使用 selector 的贪心动作，避免把探索噪声直接写入负载反馈。
        expert_load = {}
        routed_load = {}

        def _accumulate_load(load_dict, sel):
            if sel is None:
                return
            if isinstance(sel, np.ndarray):
                sel = sel.astype(np.int64).tolist()
            if isinstance(sel, (list, tuple)):
                for g, e in enumerate(list(sel)):
                    key = f"g{int(g)}_e{int(e)}"
                    load_dict[key] = load_dict.get(key, 0) + 1
            else:
                load_dict[int(sel)] = load_dict.get(int(sel), 0) + 1

        for client_proxy, info in zip(selected_clients, client_mean_gates_list):
            routed_sel = info.get("selected_expert", None) if isinstance(info, dict) else None
            _accumulate_load(routed_load, routed_sel)

            greedy_sel = None
            try:
                client_obj = getattr(client_proxy, "client", None)
                selector = getattr(client_obj, "selector", None) if client_obj is not None else None
                if selector is not None and hasattr(selector, "select_greedy") and hasattr(client_obj, "_collect_state"):
                    state = client_obj._collect_state()
                    greedy_sel = selector.select_greedy(state)
            except Exception:
                greedy_sel = None

            _accumulate_load(expert_load, greedy_sel if greedy_sel is not None else routed_sel)

        self.global_model.expert_load = expert_load
        log(INFO, f"[Aggregate] Expert load distribution (for next-round penalty): {expert_load}")
        log(INFO, f"[Aggregate] Expert routed load (train-time actions): {routed_load}")






        return self.global_model
    
    

    def aggregate_models2(self, fl_round: int, results: List[Tuple[List[np.ndarray], int]], history: History) -> torch.nn.Module:
        log(INFO, f"[Global round {fl_round}] Aggregating local models using DDPG...")

        # 收集客户端状态特征
        client_features = []
        for client_id, (weights, num_samples) in enumerate(results):
            client_model_weights = self.client_proxies[client_id].get_parameters()
            _, loss, metrics = self.client_proxies[client_id].evaluate(method="test")
            mse = metrics["MSE"]
            rmse = metrics["RMSE"]
            mae = metrics["MAE"]
            r2 = metrics["R^2"]
            nrmse = metrics["NRMSE"]
            data_ratio = num_samples / sum([r[1] for r in results])
            # 定义 client_data 为客户端模型参数的展平数组
            client_data = np.concatenate([param.flatten() for param in client_model_weights])            #新增
            mean = np.mean(client_data)
            std = np.std(client_data)
            skewness = scipy.stats.skew(client_data)
            kurtosis = scipy.stats.kurtosis(client_data)

            client_features.append([mse, rmse, mae, r2, nrmse, data_ratio,mean, std, skewness, kurtosis])
            

        # client_models = [self.client_proxies[client_id].get_parameters() for client_id in range(len(self.client_proxies))]
        client_models = []
        for client_id in range(len(self.client_proxies)):
            model = copy.deepcopy(self.global_model)  # 创建一个新的模型对象
            client_model_params = self.client_proxies[client_id].get_parameters()  # 获取客户端参数
            model.load_state_dict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), client_model_params)})  # 加载参数
            client_models.append(model)  # 将模型添加到列表中
        
        global_model = self.global_model
        new_global_w, client_weights = aggregate_with_ddpg(
            self.ddpg_agent,
            client_features,
            client_models,
            global_model,
            previous_weights=self.previous_weights  # 传递 previous_weights
        )
        self.previous_weights = client_weights  # 更新 previous_weights
        log(INFO, f"Client weights for this round: {client_weights}")


        # === 新增：下发 RL 权重给每个客户端 ===
        base_contrastive_lambda = 0.0005  # 你可以根据需要调整
        for cid, client_proxy in enumerate(self.client_proxies):
            # 只有ddpg聚合时才下发动态权重
            if args.aggregation == "ddpg":
                contrastive_lambda = base_contrastive_lambda * client_weights[cid]
            else:
                contrastive_lambda = base_contrastive_lambda
            client_proxy.set_train_parameters({'contrastive_lambda': float(contrastive_lambda)})






        # 更新 global_model
        global_model = copy.deepcopy(self.global_model)
        global_model.load_state_dict(new_global_w)


        # 定义 reward
        previous_global_loss = history.global_test_losses[-1] if len(history.global_test_losses) > 0 else np.inf
        previous_val_r2 = history.global_test_metrics["R^2"][-1] if len(history.global_test_metrics["R^2"]) > 0 else -np.inf

        self.global_model = global_model  # 更新全局模型
        self.evaluate_round(fl_round, history)  # 使用 evaluate_round 方法评估全局模型
        current_global_loss = history.global_test_losses[-1]  # 从 history 中获取最新的测试损失
        current_val_r2 = history.global_test_metrics["R^2"][-1]

        # reward = previous_global_loss - current_global_loss
        reward = (previous_global_loss - current_global_loss) + 0.5 * (current_val_r2 - previous_val_r2)


        # 定义 action 和 next_state
        action = client_weights  # 动作是客户端权重



        # next_state = client_features
        # next_state = self.calculate_next_state(global_model, client_features)
        next_state = self.calculate_next_state(global_model, results)  # 基于聚合后的全局模型测试结果生成下一状态


        client_features = np.array(client_features).mean(axis=0)  # 将维度从 [num_clients, state_dim] 转换为 [state_dim]


        # 存储经验
        self.ddpg_agent.save_experience(client_features, action, reward, next_state)

        # 更新强化学习模型
        self.ddpg_agent.update()


        return global_model



    def _collect_greedy_experts_for_current_round(self):
        selected_experts = {}

        for cp in getattr(self, "client_proxies", []):
            cid = getattr(cp, "cid", None)
            client_obj = getattr(cp, "client", cp)

            greedy_expert = None

            selector = getattr(client_obj, "selector", None)
            collect_state_fn = getattr(client_obj, "_collect_state", None)

            if selector is not None and callable(collect_state_fn):
                try:
                    state = collect_state_fn()
                    if hasattr(selector, "select_greedy"):
                        greedy_expert = selector.select_greedy(state)
                        if isinstance(greedy_expert, np.ndarray):
                            greedy_expert = greedy_expert.astype(np.int64).tolist()
                    else:
                        state_tensor = torch.tensor(
                            state, dtype=torch.float32, device=selector.device
                        ).unsqueeze(0)
                        with torch.no_grad():
                            greedy_expert = int(selector.net(state_tensor).argmax().item())
                except Exception as e:
                    log(INFO, f"[{cid}] failed to collect greedy expert at current round: {e}")
                    greedy_expert = None

            if greedy_expert is None:
                greedy_expert = getattr(cp, "selected_expert", None)
            if greedy_expert is None:
                greedy_expert = getattr(client_obj, "selected_expert", None)

            selected_experts[cid] = greedy_expert

        return selected_experts


    def calculate_next_state(self, global_model, results):
        """基于聚合后的全局模型测试结果生成下一状态"""
        next_state = []
        for client_id, (weights, num_samples) in enumerate(results):
            _, loss, metrics = self.client_proxies[client_id].evaluate(model=global_model, method="test")
            mse = metrics["MSE"]
            rmse = metrics["RMSE"]
            mae = metrics["MAE"]
            r2 = metrics["R^2"]
            nrmse = metrics["NRMSE"]
            data_ratio = num_samples / sum([r[1] for r in results])

              # 获取客户端模型参数并计算新增指标
            client_model_weights = self.client_proxies[client_id].get_parameters()
            client_data = np.concatenate([param.flatten() for param in client_model_weights])
            mean = np.mean(client_data)
            std = np.std(client_data)
            skewness = scipy.stats.skew(client_data)
            kurtosis = scipy.stats.kurtosis(client_data)


            next_state.append([mse, rmse, mae, r2, nrmse, data_ratio, mean, std, skewness, kurtosis])  # 将测试结果作为下一状态
        return np.array(next_state).mean(axis=0)  # 将维度从 [num_clients, state_dim] 转换为 [state_dim]


    def evaluate_round(self, fl_round: int, history: History):
        """Evaluate global model."""
        num_train_examples: List[int] = []
        train_losses: Dict[str, float] = dict()
        train_metrics: Dict[str, Dict[str, float]] = dict()
        num_test_examples: List[int] = []
        test_losses: Dict[str, float] = dict()
        test_metrics: Dict[str, Dict[str, float]] = dict()

        if fl_round == 0:
            #log(INFO, "Evaluating initial global model")
            self.global_model: List[np.ndarray] = self._get_initial_model()

        for cid, client_proxy in self.client_manager.all().items():
            if not self.val_loader:
                num_train_instances, train_loss, train_eval_metrics = client_proxy.evaluate(
                    model=self.global_model, method="train"
                )
                num_train_examples.append(num_train_instances)
                train_losses[cid] = train_loss
                train_metrics[cid] = train_eval_metrics

            # Always use each client's own validation split to avoid shared-loader skew.
            num_test_instances, test_loss, test_eval_metrics = client_proxy.evaluate(
                model=self.global_model, method="test"
            )
            num_test_examples.append(num_test_instances)
            test_losses[cid] = test_loss
            test_metrics[cid] = test_eval_metrics

        if len(num_train_examples) > 0:
            history.add_global_train_losses(self.weighted_loss(num_train_examples, list(train_losses.values())))
            history.add_global_train_metrics(self.weighted_metrics(num_train_examples, train_metrics))

        history.add_global_test_losses(self.weighted_loss(num_test_examples, list(test_losses.values())))
        # if history.global_test_losses[-1] <= self.best_loss:
        #     #log(DEBUG, f"Caching best global model, fl_round={fl_round}")
        #     self.best_loss = history.global_test_losses[-1]
        #     self.best_epoch = fl_round
        #     self.best_model = copy.deepcopy(self.global_model)
        current_loss = float(history.global_test_losses[-1])
        min_delta = float(getattr(args, "best_model_min_delta", 1e-6))

        if self.best_model is None or current_loss < (self.best_loss - min_delta):
            self.best_loss = current_loss
            self.best_epoch = fl_round
            self.best_model = copy.deepcopy(self.global_model)
            self.no_improve_rounds = 0

            self.best_selected_experts_by_client = self._collect_greedy_experts_for_current_round()
            self.best_model.selected_experts_by_client = copy.deepcopy(self.best_selected_experts_by_client)

            log(INFO, f"[Best Round {fl_round}] cached greedy selected_experts_by_client: {self.best_selected_experts_by_client}")
        else:
            self.no_improve_rounds += 1
        history.add_global_test_metrics(self.weighted_metrics(num_test_examples, test_metrics))

    # def _get_initial_model(self) -> List[np.ndarray]:
    #     """Get initial parameters from a random client"""
    #     random_client = self.client_manager.sample(0.)[0]
    #     client_model = random_client.get_parameters()
    #     # log(INFO, "Received initial parameters from one random client!")
    #     return client_model
    def _get_initial_model(self) -> torch.nn.Module:
        """Get initial parameters from a random client"""
        random_client = self.client_manager.sample(0.)[0]
        client_model_params = random_client.get_parameters()
        # 确保 self.global_model 已正确初始化
        if self.global_model is None:
            raise ValueError("self.global_model 未正确初始化，请检查 Server 类的初始化逻辑。")

        model = copy.deepcopy(self.global_model)
        model_state = model.state_dict()
        load_dict = {}
        mismatches = []

        for key, value in zip(model_state.keys(), client_model_params):
            tensor = torch.tensor(value)
            if tensor.shape == model_state[key].shape:
                load_dict[key] = tensor
            else:
                mismatches.append((key, tuple(tensor.shape), tuple(model_state[key].shape)))

        if mismatches:
            log(INFO, f"[Init] skip {len(mismatches)} mismatched params when loading initial model")

        if load_dict:
            model.load_state_dict(load_dict, strict=False)
        else:
            log(INFO, "[Init] no compatible params from client, using fresh global model")

        return model
