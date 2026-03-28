import copy
import random
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from flcore.clients.clientHGFIDSUS import clientHGFIDSUS
from flcore.servers.serverbase import Server


class LogitAdjustedCE(nn.Module):
    def __init__(self, num_classes: int, tau: float = 1.0, ema: float = 0.99):
        super().__init__()
        self.tau = float(tau)
        self.ema = float(max(0.0, min(0.9999, ema)))
        self.register_buffer("count_ema", torch.ones(num_classes))

    @torch.no_grad()
    def update_count(self, y: torch.Tensor):
        if y.numel() == 0:
            return
        binc = torch.bincount(y.view(-1), minlength=self.count_ema.numel()).float().to(self.count_ema.device)
        self.count_ema.mul_(self.ema).add_(binc, alpha=(1.0 - self.ema))

    def forward(self, logits: torch.Tensor, y: torch.Tensor):
        prior = self.count_ema / torch.clamp_min(torch.sum(self.count_ema), 1e-12)
        adj = self.tau * torch.log(prior + 1e-12)
        return nn.functional.cross_entropy(logits + adj.unsqueeze(0), y)


class HGFIDSUS(Server):
    """Hierarchical-Game FIDSUS with paper-aligned updates.

    Key additions:
    - entropy / credibility aware selection inspired by hybrid cyber-defense work
    - split-style personalization and failover-inspired robustness for UAV dynamics
    - cleaner metric set with calibration / robustness / efficiency indicators
    - optional early stopping to control total training time
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients(clientHGFIDSUS)

        self.P = torch.diag(torch.ones(self.num_clients, device=self.device))
        self.uploaded_ids = []
        self.M = min(args.M, self.num_join_clients)
        self.client_models = [copy.deepcopy(self.global_model) for _ in range(self.num_clients)]

        self.Budget = []
        self.CEloss = nn.CrossEntropyLoss()

        self.head = self.client_models[0].head
        self.opt_h = torch.optim.SGD(self.head.parameters(), lr=self.server_learning_rate)
        self.la_tau = float(getattr(args, "la_tau", 1.0))
        self.head_criterion = LogitAdjustedCE(self.num_classes, tau=self.la_tau).to(self.device)

        self.reputation = torch.ones(self.num_clients, device=self.device)
        self.staleness = torch.zeros(self.num_clients, dtype=torch.long, device=self.device)

        self.price_compute = float(getattr(args, "price_compute", 1.0))
        self.price_comm = float(getattr(args, "price_comm", 1.0))
        self.dynamic_pricing = bool(getattr(args, "dynamic_pricing", True))
        self.price_step = float(getattr(args, "price_step", 0.05))
        self.price_compute_min = float(getattr(args, "price_compute_min", 0.7))
        self.price_compute_max = float(getattr(args, "price_compute_max", 2.2))
        self.price_comm_min = float(getattr(args, "price_comm_min", 0.7))
        self.price_comm_max = float(getattr(args, "price_comm_max", 2.2))
        self.price_time_tolerance = float(getattr(args, "price_time_tolerance", 0.05))
        self.price_target_round_time = float(getattr(args, "price_target_round_time", 0.0))
        self.price_warmup_rounds = int(getattr(args, "price_warmup_rounds", 5))
        self.price_acc_drop_th = float(getattr(args, "price_acc_drop_th", 0.01))
        self.price_acc_gain_th = float(getattr(args, "price_acc_gain_th", 0.01))
        self.price_staleness_target = float(getattr(args, "price_staleness_target", 3.0))
        self.price_entropy_relax_th = float(getattr(args, "price_entropy_relax_th", 0.60))
        self._round_time_ema = None
        self._warmup_round_times: List[float] = []

        self.bandwidth_budget = int(getattr(args, "bandwidth_budget", 0))
        self.auction_winners_frac = float(getattr(args, "auction_winners_frac", 0.6))
        self.delta_min_norm = float(getattr(args, "delta_min_norm", 1e-15))
        self.delta_server_lr = float(getattr(args, "delta_server_lr", 0.5))
        self.delta_clip_norm = float(getattr(args, "delta_clip_norm", 0.0))
        self.delta_trim_ratio = float(getattr(args, "delta_trim_ratio", 0.1))
        self.warmup_rounds = int(getattr(args, "warmup_rounds", 8))
        self.current_round = 0
        self.global_base_alpha = float(getattr(args, "global_base_alpha", 0.05))
        self.max_staleness_upload = int(getattr(args, "max_staleness_upload", 0))
        self.stale_priority_th = int(getattr(args, "stale_priority_th", 4))
        self.stale_quota_frac = float(getattr(args, "stale_quota_frac", 0.3))
        self.stale_score_alpha = float(getattr(args, "stale_score_alpha", 0.25))
        self.stale_score_cap = int(getattr(args, "stale_score_cap", 30))
        self.head_train_epochs = int(getattr(args, "head_train_epochs", 2))
        self.min_delta_uploaders = int(getattr(args, "min_delta_uploaders", 2))
        self.staleness_decay_gamma = float(getattr(args, "staleness_decay_gamma", 0.10))

        self.enable_contracts = bool(getattr(args, "enable_contracts", True))
        self.contract_start_round = int(getattr(args, "contract_start_round", self.warmup_rounds))
        self.min_full_clients = int(getattr(args, "min_full_clients", 4))
        self.contract_compute_discount = float(getattr(args, "contract_compute_discount", 0.25))
        self.contract_comm_discount = float(getattr(args, "contract_comm_discount", 0.20))
        self.contract_bonus = float(getattr(args, "contract_bonus", 0.10))
        self.contract_clients = set()

        self.enable_sticky_sampling = bool(getattr(args, "enable_sticky_sampling", True))
        self.sticky_fraction = float(getattr(args, "sticky_fraction", 0.50))
        self.no_replace_window = int(getattr(args, "no_replace_window", 3))
        self._prev_selected_ids: List[int] = []
        self._recent_selected_windows = deque(maxlen=max(self.no_replace_window, 1))

        self.enable_delta_compression = bool(getattr(args, "enable_delta_compression", True))
        self.proto_fp16 = bool(getattr(args, "proto_fp16", True))

        self.use_failover = bool(getattr(args, "use_failover", True))
        self.failover_decay = float(getattr(args, "failover_decay", 0.75))
        self.failover_max_age = int(getattr(args, "failover_max_age", 8))
        self.delta_cache: Dict[int, Dict[str, object]] = {}

        self.force_affinity_drop_rounds = int(getattr(args, "force_affinity_drop_rounds", 1))
        self.force_affinity_drop_threshold = float(getattr(args, "force_affinity_drop_threshold", 0.02))
        self.eval_acc_ema_alpha = float(getattr(args, "eval_acc_ema_alpha", 0.2))
        self.eval_calibration = bool(getattr(args, "eval_calibration", True))
        self.eval_calibration_start_round = int(getattr(args, "eval_calibration_start_round", 0))
        self.eval_calibration_temp_min = float(getattr(args, "eval_calibration_temp_min", 0.6))
        self.eval_calibration_temp_max = float(getattr(args, "eval_calibration_temp_max", 3.0))
        self.eval_calibration_temp_steps = int(getattr(args, "eval_calibration_temp_steps", 25))
        self._recent_eval_acc: List[float] = []
        self._eval_acc_ema = None

        self.early_stop_patience = int(getattr(args, "early_stop_patience", 0))
        self.early_stop_min_rounds = int(getattr(args, "early_stop_min_rounds", 0))
        self.checkpoint_mode = str(getattr(args, "checkpoint_mode", "multi")).strip().lower()
        self.checkpoint_w_acc = float(getattr(args, "checkpoint_w_acc", 0.35))
        self.checkpoint_w_macro_f1 = float(getattr(args, "checkpoint_w_macro_f1", 0.25))
        self.checkpoint_w_balanced_acc = float(getattr(args, "checkpoint_w_balanced_acc", 0.20))
        self.checkpoint_w_few_shot_macro_f1 = float(getattr(args, "checkpoint_w_few_shot_macro_f1", 0.10))
        self.checkpoint_w_ece = float(getattr(args, "checkpoint_w_ece", 0.05))
        self.checkpoint_w_brier = float(getattr(args, "checkpoint_w_brier", 0.05))
        self._best_eval_acc = -1.0
        self._best_eval_score = -float("inf")
        self._best_eval_round = -1
        self._no_improve_evals = 0
        self._best_global_state = None
        self._best_head_state = None

        # Alignment keys (per-round)
        self.rs_round_idx = []
        self.rs_eval_flag = []
        self.rs_force_affinity_flag = []

        # Common comparison fields (per-round)
        self.rs_selected_clients = []
        self.rs_active_ratio = []
        self.rs_heavy_uploader_clients = []
        self.rs_heavy_uploader_ratio = []
        self.rs_upload_success_ratio = []
        self.rs_upload_per_winner = []
        self.rs_drop_count = []
        self.rs_bandwidth_budget_bytes = []
        self.rs_bandwidth_utilization = []

        # HGF-mechanism introspection (per-round)
        self.rs_mode_full_ratio = []
        self.rs_mode_freeze_base_ratio = []
        self.rs_mode_head_only_ratio = []
        self.rs_proto_drift_mean = []
        self.rs_bid_loss_drop_mean = []
        self.rs_bid_drift_mean = []
        self.rs_bid_staleness_mean = []
        self.rs_reject_time_count = []
        self.rs_reject_missing_delta_count = []
        self.rs_reject_below_norm_count = []
        self.rs_reject_total_count = []
        self.rs_mean_pred_entropy = []
        self.rs_mean_credibility = []
        self.rs_failover_used_count = []
        self.rs_failover_round_flag = []
        self.rs_contract_clients = []
        self.rs_contract_active_ratio = []
        self.rs_contract_winner_ratio = []
        self.rs_mean_delta_sparsity = []

        self.rs_round_time = []
        self.rs_test_acc_ema = []
        self.rs_test_macro_f1 = []
        self.rs_test_weighted_f1 = []
        self.rs_test_macro_recall = []  # kept for backward compatibility; not printed because redundant
        self.rs_test_balanced_acc = []
        self.rs_test_macro_auc = []
        self.rs_test_macro_prauc = []
        self.rs_test_mcc = []
        self.rs_test_kappa = []
        self.rs_test_ece = []
        self.rs_test_brier = []
        self.rs_test_micro_fpr = []
        self.rs_test_avg_conf = []
        self.rs_test_ece_raw = []
        self.rs_test_brier_raw = []
        self.rs_test_avg_conf_raw = []
        self.rs_eval_calibration_temp = []
        self.rs_eval_calibration_samples = []
        self.rs_selection_score = []
        self.rs_best_selection_score = []
        self.rs_checkpoint_improved = []
        self.rs_active_clients = []
        self.rs_winner_clients = []
        self.rs_winner_ratio = []
        self.rs_affinity_update_ratio = []
        self.rs_proto_count = []
        self.rs_proto_upload_bytes = []
        self.rs_delta_upload_bytes = []
        self.rs_total_upload_bytes = []
        self.rs_mean_delta_norm = []
        self.rs_mean_bid_value = []
        self.rs_avg_client_train_time = []
        self.rs_avg_client_send_time = []
        self.rs_reputation_mean = []
        self.rs_reputation_std = []
        self.rs_staleness_mean = []
        self.rs_staleness_max = []
        self.rs_price_compute = []
        self.rs_price_comm = []

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        stop_early = False
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.current_round = i

            self.rs_round_idx.append(float(i))
            self.rs_eval_flag.append(float(1.0 if (i % self.eval_gap == 0) else 0.0))
            self.rs_force_affinity_flag.append(float(1.0 if getattr(self, "_force_affinity", False) else 0.0))

            self.selected_clients = self.select_clients()
            self._assign_stackelberg_contracts()
            self._broadcast_round_context(round_idx=i)
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate_personalized()
                self._update_force_affinity()
                stop_early = self._maybe_early_stop(i)
                if stop_early:
                    print(f"Early stop triggered at round {i}; restoring best checkpoint from round {self._best_eval_round}.")
                    self._restore_best_checkpoint()
                    break

            for client in self.selected_clients:
                client.train()

            self.receive_models(round_idx=i)
            if i >= self.warmup_rounds:
                self._aggregate_base_deltas()
            self.train_head()
            self._sync_global_head()

            round_time = time.time() - s_t
            self.Budget.append(round_time)
            self.rs_round_time.append(float(round_time))
            self._update_dynamic_prices(round_time)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])
            self._print_round_resource_metrics()

        print("\nBest accuracy.")
        print(max(self.rs_test_acc) if len(self.rs_test_acc) else None)
        print("\nAverage time cost per round.")
        if len(self.Budget) > 1:
            print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        elif len(self.Budget) == 1:
            print(self.Budget[0])
        self.save_results()

    def _broadcast_round_context(self, round_idx: int):
        force_affinity = getattr(self, "_force_affinity", False)
        mean_field_drift = float(self.rs_proto_drift_mean[-1]) if len(self.rs_proto_drift_mean) > 0 else 0.0
        for client in self.clients:
            is_contract = client.id in self.contract_clients
            client.server_price_compute = self.price_compute
            client.server_price_comm = self.price_comm
            client.force_affinity_update = force_affinity
            client.round_idx = round_idx
            client.server_reputation = float(self.reputation[client.id].item())
            client.server_staleness = int(self.staleness[client.id].item())
            client.server_mean_field_drift = mean_field_drift
            client.server_contract_mode = "full" if is_contract else ""
            client.server_contract_bonus = self.contract_bonus if is_contract else 0.0
            client.server_contract_discount_compute = self.contract_compute_discount if is_contract else 0.0
            client.server_contract_discount_comm = self.contract_comm_discount if is_contract else 0.0

    def _sync_global_head(self):
        for dst, src in zip(self.global_model.head.parameters(), self.head.parameters()):
            dst.data.copy_(src.data)

    def _update_force_affinity(self):
        if len(self.rs_test_acc_ema) == 0:
            return
        wnd = max(2, self.force_affinity_drop_rounds)
        self._recent_eval_acc.append(self.rs_test_acc_ema[-1])
        if len(self._recent_eval_acc) > wnd:
            self._recent_eval_acc = self._recent_eval_acc[-wnd:]

        if len(self._recent_eval_acc) < 2:
            self._force_affinity = False
            return

        drops = []
        for j in range(1, len(self._recent_eval_acc)):
            prev = self._recent_eval_acc[j - 1]
            cur = self._recent_eval_acc[j]
            drops.append((prev - cur) > self.force_affinity_drop_threshold)

        self._force_affinity = all(drops)

    def _snapshot_best_checkpoint(self):
        self._best_global_state = _clone_state_dict(self.global_model.state_dict())
        self._best_head_state = _clone_state_dict(self.head.state_dict())

    def _restore_best_checkpoint(self):
        if self._best_global_state is not None:
            self.global_model.load_state_dict(self._best_global_state, strict=True)
        if self._best_head_state is not None:
            self.head.load_state_dict(self._best_head_state, strict=True)
            self._sync_global_head()

    def _maybe_early_stop(self, round_idx: int) -> bool:
        if self.early_stop_patience <= 0:
            return False
        if round_idx < self.early_stop_min_rounds:
            return False
        return self._no_improve_evals >= self.early_stop_patience

    def _selection_metric_name(self) -> str:
        if self.checkpoint_mode in {"acc", "accuracy"}:
            return "accuracy"
        return "multi_objective"

    def _compute_selection_score(self, metrics: Dict[str, float]) -> float:
        if self.checkpoint_mode in {"acc", "accuracy"}:
            return float(metrics.get("test_acc", 0.0))

        return float(
            self.checkpoint_w_acc * float(metrics.get("test_acc", 0.0))
            + self.checkpoint_w_macro_f1 * float(metrics.get("macro_f1", 0.0))
            + self.checkpoint_w_balanced_acc * float(metrics.get("balanced_acc", 0.0))
            + self.checkpoint_w_few_shot_macro_f1 * float(metrics.get("few_shot_macro_f1", 0.0))
            - self.checkpoint_w_ece * float(metrics.get("ece", 0.0))
            - self.checkpoint_w_brier * float(metrics.get("brier", 0.0))
        )

    def _update_checkpoint_state(self, metrics: Dict[str, float]):
        score = self._compute_selection_score(metrics)
        improved = 0.0
        if score > self._best_eval_score + 1e-8:
            self._best_eval_score = float(score)
            self._best_eval_acc = float(metrics.get("test_acc", self._best_eval_acc))
            self._best_eval_round = int(self.current_round)
            self._no_improve_evals = 0
            improved = 1.0
            self._snapshot_best_checkpoint()
        else:
            self._no_improve_evals += 1

        self.rs_selection_score.append(float(score))
        self.rs_best_selection_score.append(float(self._best_eval_score))
        self.rs_checkpoint_improved.append(float(improved))

        print("Selection Score ({}): {:.4f}".format(self._selection_metric_name(), score))
        if improved > 0.0:
            print("Checkpoint Updated: round {}".format(int(self.current_round)))

    def _after_personalized_eval(
        self,
        metrics: Dict[str, float],
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_prob_cal: Optional[np.ndarray] = None,
    ):
        self._update_checkpoint_state(metrics)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(
                range(self.num_join_clients, self.num_clients + 1), 1, replace=False
            )[0]
        else:
            self.current_num_join_clients = self.num_join_clients

        target = int(max(1, self.current_num_join_clients))
        if (not self.enable_sticky_sampling) or self.current_round <= 0:
            selected_ids = random.sample(list(range(self.num_clients)), target)
        else:
            sticky_target = int(round(target * float(np.clip(self.sticky_fraction, 0.0, 1.0))))
            sticky_target = min(sticky_target, len(self._prev_selected_ids), target)
            sticky_ids = random.sample(self._prev_selected_ids, sticky_target) if sticky_target > 0 else []

            blocked = set()
            if self.no_replace_window > 0:
                for item in self._recent_selected_windows:
                    blocked.update(item)
            blocked.difference_update(sticky_ids)

            selected_ids = list(sticky_ids)
            remain = target - len(selected_ids)
            fresh_pool = [cid for cid in range(self.num_clients) if cid not in selected_ids and cid not in blocked]
            if len(fresh_pool) < remain:
                fresh_pool = [cid for cid in range(self.num_clients) if cid not in selected_ids]
            if remain > 0 and len(fresh_pool) > 0:
                selected_ids.extend(random.sample(fresh_pool, min(remain, len(fresh_pool))))

            if len(selected_ids) < target:
                fallback = [cid for cid in range(self.num_clients) if cid not in selected_ids]
                if len(fallback) > 0:
                    selected_ids.extend(random.sample(fallback, target - len(selected_ids)))

        self._prev_selected_ids = list(selected_ids)
        if self.no_replace_window > 0:
            self._recent_selected_windows.append(set(selected_ids))
        return [self.clients[cid] for cid in selected_ids]

    def _assign_stackelberg_contracts(self):
        self.contract_clients = set()
        if (not self.enable_contracts) or self.current_round < self.contract_start_round:
            return

        candidate_scores = []
        for client in self.selected_clients:
            quality = float(getattr(client, "quality_score", 0.0))
            credibility = float(getattr(client, "credibility_score", 0.0))
            delta_quality = float(getattr(client, "delta_quality", 0.0))
            stale = float(self.staleness[client.id].item())
            rep = float(self.reputation[client.id].item())
            contract_score = 0.55 * credibility + 0.25 * quality + 0.10 * delta_quality + 0.05 * stale + 0.05 * rep
            candidate_scores.append((contract_score, client.id))

        candidate_scores.sort(reverse=True)
        quota = min(max(0, self.min_full_clients), len(candidate_scores), max(1, self.current_num_join_clients))
        self.contract_clients = {cid for _, cid in candidate_scores[:quota]}

    def send_models(self):
        assert len(self.selected_clients) > 0
        for client in self.selected_clients:
            start_time = time.time()

            m_ = min(self.M, len(self.uploaded_ids))
            if m_ <= 0:
                indices = []
            else:
                indices = torch.topk(self.P[client.id], m_).indices.tolist()

            send_ids = []
            send_models = []
            for idx in indices:
                send_ids.append(idx)
                send_models.append(self.client_models[idx])

            client.receive_models(send_ids, send_models)
            client.set_parameters(self.head)
            client.set_global_base(self.global_model.base, alpha=self.global_base_alpha)

            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)

    def _cache_client_delta(self, client):
        if getattr(client, "delta_base", None) is None:
            return
        self.delta_cache[int(client.id)] = {
            "delta": [d.detach().cpu().clone() for d in client.delta_base],
            "round": int(self.current_round),
            "norm": float(getattr(client, "delta_norm", 0.0)),
            "size": int(getattr(client, "delta_size", 0)),
            "credibility": float(getattr(client, "credibility_score", 0.0)),
        }

    def _get_failover_delta(self, client_id: int, round_idx: int):
        if not self.use_failover:
            return None
        info = self.delta_cache.get(int(client_id))
        if info is None:
            return None
        age = int(round_idx) - int(info.get("round", round_idx))
        if age < 0 or age > self.failover_max_age:
            return None
        scale = float(self.failover_decay) ** age
        base_delta = info.get("delta", None)
        if base_delta is None:
            return None
        delta = [d.to(self.device).clone() * scale for d in base_delta]
        norm = float(info.get("norm", 0.0)) * scale
        size = int(info.get("size", 0))
        cred = float(info.get("credibility", 0.0))
        if norm < self.delta_min_norm:
            return None
        return delta, norm, size, cred

    def _fill_failover_pool(self, round_idx: int, need: int, exclude_ids: Optional[set] = None):
        if not self.use_failover or need <= 0:
            return []
        exclude_ids = set() if exclude_ids is None else set(exclude_ids)
        candidates = []
        for cid, info in self.delta_cache.items():
            if cid in exclude_ids:
                continue
            age = int(round_idx) - int(info.get("round", round_idx))
            if age < 0 or age > self.failover_max_age:
                continue
            candidates.append((float(info.get("credibility", 0.0)), -age, int(cid)))
        candidates.sort(reverse=True)
        selected = []
        for _, _, cid in candidates[:need]:
            item = self._get_failover_delta(cid, round_idx)
            if item is not None:
                selected.append((cid, item))
        return selected

    def receive_models(self, round_idx: int):
        assert len(self.selected_clients) > 0

        active_clients = random.sample(
            self.selected_clients, int(self.client_activity_rate * self.current_num_join_clients)
        )

        winners = self._select_auction_winners(active_clients)
        winner_ids = {c.id for c in winners}

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_protos = []
        self.uploaded_deltas: List[List[torch.Tensor]] = []
        self.uploaded_delta_sizes: List[int] = []

        active_train_times = []
        active_send_times = []
        affinity_update_count = 0
        bid_values = []
        delta_norm_values = []
        proto_drift_values = []
        bid_loss_drop_values = []
        bid_drift_values = []
        bid_staleness_values = []
        pred_entropy_values = []
        credibility_values = []
        delta_sparsity_values = []
        mode_full = 0
        mode_freeze = 0
        mode_head = 0
        failover_used = 0
        contract_active_count = 0
        contract_uploaded_count = 0

        reject_time = 0
        reject_missing_delta = 0
        reject_below_norm = 0

        tot_samples = 0
        for client in active_clients:
            bid_values.append(float(getattr(client, "bid_value", 0.0)))
            proto_drift_values.append(float(getattr(client, "proto_drift", 0.0)))
            bid_loss_drop_values.append(float(getattr(client, "bid_loss_drop", 0.0)))
            bid_drift_values.append(float(getattr(client, "bid_proto_drift", getattr(client, "proto_drift", 0.0))))
            bid_staleness_values.append(float(getattr(client, "bid_staleness_term", 0.0)))
            pred_entropy_values.append(float(getattr(client, "pred_entropy", 0.0)))
            credibility_values.append(float(getattr(client, "credibility_score", 0.0)))
            delta_sparsity_values.append(float(getattr(client, "delta_sparsity", 0.0)))
            if client.id in self.contract_clients:
                contract_active_count += 1

            mode = str(getattr(client, "last_train_mode", "full"))
            if mode == "head_only":
                mode_head += 1
            elif mode == "freeze_base":
                mode_freeze += 1
            else:
                mode_full += 1

            if client.train_time_cost["num_rounds"] > 0:
                active_train_times.append(client.train_time_cost["total_cost"] / client.train_time_cost["num_rounds"])
            if client.send_time_cost["num_rounds"] > 0:
                active_send_times.append(client.send_time_cost["total_cost"] / client.send_time_cost["num_rounds"])

            for cc in client.protos.keys():
                y = torch.tensor(cc, dtype=torch.int64)
                self.uploaded_protos.append((client.protos[cc], y))

            if getattr(client, "affinity_updated", True):
                affinity_update_count += 1
                scale = float(self.reputation[client.id].clamp(min=0.1, max=2.0).item())
                self.P[client.id] += client.weight_vector * scale

            self.client_models[client.id] = copy.deepcopy(client.model)
            self._cache_client_delta(client)

            if client.id in winner_ids:
                try:
                    client_time_cost = (
                        client.train_time_cost["total_cost"] / client.train_time_cost["num_rounds"]
                        + client.send_time_cost["total_cost"] / client.send_time_cost["num_rounds"]
                    )
                except ZeroDivisionError:
                    client_time_cost = 0

                failover_item = None
                if client_time_cost > self.time_threthold:
                    reject_time += 1
                    failover_item = self._get_failover_delta(client.id, round_idx)
                elif getattr(client, "delta_base", None) is None:
                    reject_missing_delta += 1
                    failover_item = self._get_failover_delta(client.id, round_idx)
                else:
                    delta_norm = float(getattr(client, "delta_norm", 0.0))
                    if delta_norm < self.delta_min_norm:
                        reject_below_norm += 1
                        failover_item = self._get_failover_delta(client.id, round_idx)
                    else:
                        tot_samples += client.train_samples
                        self.uploaded_ids.append(client.id)
                        self.uploaded_weights.append(client.train_samples)
                        self.uploaded_deltas.append(client.delta_base)
                        self.uploaded_delta_sizes.append(int(getattr(client, "delta_size", 0)))
                        delta_norm_values.append(delta_norm)
                        self.staleness[client.id] = 0
                        self._reward_reputation(client.id, good=True)
                        if client.id in self.contract_clients:
                            contract_uploaded_count += 1
                        continue

                if failover_item is not None:
                    delta, delta_norm, delta_size, _ = failover_item
                    tot_samples += client.train_samples
                    self.uploaded_ids.append(client.id)
                    self.uploaded_weights.append(client.train_samples)
                    self.uploaded_deltas.append(delta)
                    self.uploaded_delta_sizes.append(int(delta_size))
                    delta_norm_values.append(float(delta_norm))
                    self.staleness[client.id] += 1
                    self._reward_reputation(client.id, good=False, mild=True)
                    failover_used += 1
                    if client.id in self.contract_clients:
                        contract_uploaded_count += 1
                else:
                    self.staleness[client.id] += 1
                    self._reward_reputation(client.id, good=False)
            else:
                self.staleness[client.id] += 1
                drift = float(getattr(client, "proto_drift", 0.0))
                if drift > float(getattr(self.args, "affinity_drift_th", 0.0)):
                    self._reward_reputation(client.id, good=False, mild=True)

        if len(self.uploaded_deltas) < self.min_delta_uploaders:
            need = self.min_delta_uploaders - len(self.uploaded_deltas)
            fallback_items = self._fill_failover_pool(
                round_idx, need=need, exclude_ids=set(self.uploaded_ids)
            )
            for cid, item in fallback_items:
                delta, delta_norm, _, _ = item
                self.uploaded_ids.append(cid)
                self.uploaded_weights.append(1.0)
                self.uploaded_deltas.append(delta)
                self.uploaded_delta_sizes.append(int(item[2]))
                delta_norm_values.append(float(delta_norm))
                failover_used += 1
                if cid in self.contract_clients:
                    contract_uploaded_count += 1

        if tot_samples > 0:
            for i, w in enumerate(self.uploaded_weights):
                self.uploaded_weights[i] = w / tot_samples
        elif len(self.uploaded_weights) > 0:
            norm = float(sum(self.uploaded_weights))
            for i, w in enumerate(self.uploaded_weights):
                self.uploaded_weights[i] = w / max(norm, 1e-12)

        proto_bytes = 0
        for p, _ in self.uploaded_protos:
            try:
                proto_bytes += int(p.numel() * p.element_size())
            except Exception:
                pass

        delta_bytes = 0
        for sz in self.uploaded_delta_sizes:
            delta_bytes += int(sz)

        active_cnt = len(active_clients)
        winner_cnt = len(winners)
        uploaded_cnt = len(self.uploaded_ids)
        selected_cnt = len(getattr(self, "selected_clients", []))

        self.rs_selected_clients.append(float(selected_cnt))
        self.rs_active_ratio.append(float(active_cnt / max(selected_cnt, 1)))
        self.rs_heavy_uploader_clients.append(float(uploaded_cnt))
        self.rs_heavy_uploader_ratio.append(float(uploaded_cnt / max(active_cnt, 1)))
        upload_per_winner = float(uploaded_cnt / max(winner_cnt, 1))
        upload_success = float(uploaded_cnt / winner_cnt) if winner_cnt > 0 else 0.0
        upload_success = float(np.clip(upload_success, 0.0, 1.0))
        self.rs_upload_success_ratio.append(upload_success)
        self.rs_upload_per_winner.append(upload_per_winner)
        self.rs_drop_count.append(float(max(winner_cnt - uploaded_cnt, 0)))
        self.rs_bandwidth_budget_bytes.append(float(self.bandwidth_budget))
        self.rs_bandwidth_utilization.append(
            float(delta_bytes / self.bandwidth_budget) if self.bandwidth_budget > 0 else 0.0
        )

        self.rs_mode_full_ratio.append(float(mode_full / max(active_cnt, 1)))
        self.rs_mode_freeze_base_ratio.append(float(mode_freeze / max(active_cnt, 1)))
        self.rs_mode_head_only_ratio.append(float(mode_head / max(active_cnt, 1)))
        self.rs_proto_drift_mean.append(float(np.mean(proto_drift_values)) if len(proto_drift_values) > 0 else 0.0)
        self.rs_bid_loss_drop_mean.append(float(np.mean(bid_loss_drop_values)) if len(bid_loss_drop_values) > 0 else 0.0)
        self.rs_bid_drift_mean.append(float(np.mean(bid_drift_values)) if len(bid_drift_values) > 0 else 0.0)
        self.rs_bid_staleness_mean.append(float(np.mean(bid_staleness_values)) if len(bid_staleness_values) > 0 else 0.0)
        self.rs_mean_pred_entropy.append(float(np.mean(pred_entropy_values)) if len(pred_entropy_values) > 0 else 0.0)
        self.rs_mean_credibility.append(float(np.mean(credibility_values)) if len(credibility_values) > 0 else 0.0)
        self.rs_failover_used_count.append(float(failover_used))
        self.rs_failover_round_flag.append(float(1.0 if failover_used > 0 else 0.0))
        self.rs_contract_clients.append(float(len(self.contract_clients)))
        self.rs_contract_active_ratio.append(float(contract_active_count / max(active_cnt, 1)))
        self.rs_contract_winner_ratio.append(float(contract_uploaded_count / max(len(self.contract_clients), 1)))
        self.rs_mean_delta_sparsity.append(float(np.mean(delta_sparsity_values)) if len(delta_sparsity_values) > 0 else 0.0)

        self.rs_reject_time_count.append(float(reject_time))
        self.rs_reject_missing_delta_count.append(float(reject_missing_delta))
        self.rs_reject_below_norm_count.append(float(reject_below_norm))
        self.rs_reject_total_count.append(float(reject_time + reject_missing_delta + reject_below_norm))

        self.rs_active_clients.append(float(active_cnt))
        self.rs_winner_clients.append(float(winner_cnt))
        self.rs_winner_ratio.append(float(winner_cnt / max(active_cnt, 1)))
        self.rs_affinity_update_ratio.append(float(affinity_update_count / max(active_cnt, 1)))
        self.rs_proto_count.append(float(len(self.uploaded_protos)))
        self.rs_proto_upload_bytes.append(float(proto_bytes))
        self.rs_delta_upload_bytes.append(float(delta_bytes))
        self.rs_total_upload_bytes.append(float(proto_bytes + delta_bytes))
        self.rs_mean_delta_norm.append(float(np.mean(delta_norm_values)) if len(delta_norm_values) > 0 else 0.0)
        self.rs_mean_bid_value.append(float(np.mean(bid_values)) if len(bid_values) > 0 else 0.0)
        self.rs_avg_client_train_time.append(float(np.mean(active_train_times)) if len(active_train_times) > 0 else 0.0)
        self.rs_avg_client_send_time.append(float(np.mean(active_send_times)) if len(active_send_times) > 0 else 0.0)
        self.rs_reputation_mean.append(float(torch.mean(self.reputation).item()))
        self.rs_reputation_std.append(float(torch.std(self.reputation, unbiased=False).item()))
        self.rs_staleness_mean.append(float(torch.mean(self.staleness.float()).item()))
        self.rs_staleness_max.append(float(torch.max(self.staleness).item()))

    def _print_round_resource_metrics(self):
        if len(self.rs_total_upload_bytes) == 0:
            return

        parts = [
            "[Round Metrics][HGFIDSUS]",
            f"selected={int(self.rs_selected_clients[-1])}",
            f"active={int(self.rs_active_clients[-1])}",
            f"winners={int(self.rs_winner_clients[-1])}",
            f"uploaded={int(self.rs_heavy_uploader_clients[-1])}",
            f"failover_used={int(self.rs_failover_used_count[-1])}",
            f"contracts={int(self.rs_contract_clients[-1])}",
            f"contract_active={self.rs_contract_active_ratio[-1]:.4f}",
            f"contract_upload={self.rs_contract_winner_ratio[-1]:.4f}",
            f"upload_success={self.rs_upload_success_ratio[-1]:.4f}",
            f"upload_per_winner={self.rs_upload_per_winner[-1]:.4f}",
            f"mode_full={self.rs_mode_full_ratio[-1]:.4f}",
            f"mode_freeze={self.rs_mode_freeze_base_ratio[-1]:.4f}",
            f"mode_head={self.rs_mode_head_only_ratio[-1]:.4f}",
            f"affinity_update_ratio={self.rs_affinity_update_ratio[-1]:.4f}",
            f"mean_delta_norm={self.rs_mean_delta_norm[-1]:.6f}",
            f"mean_delta_sparsity={self.rs_mean_delta_sparsity[-1]:.4f}",
            f"mean_bid={self.rs_mean_bid_value[-1]:.6f}",
            f"mean_entropy={self.rs_mean_pred_entropy[-1]:.4f}",
            f"mean_cred={self.rs_mean_credibility[-1]:.4f}",
            f"rep_mean={self.rs_reputation_mean[-1]:.4f}",
            f"staleness_mean={self.rs_staleness_mean[-1]:.4f}",
            f"staleness_max={int(self.rs_staleness_max[-1])}",
            f"price_compute={self.price_compute:.3f}",
            f"price_comm={self.price_comm:.3f}",
            f"avg_train_time={self.rs_avg_client_train_time[-1]:.4f}s",
            f"avg_send_time={self.rs_avg_client_send_time[-1]:.4f}s",
            f"total_upload_bytes={int(self.rs_total_upload_bytes[-1])}",
        ]
        if len(self.rs_selection_score) > 0:
            parts.append(f"selection_score={self.rs_selection_score[-1]:.4f}")
        if self.bandwidth_budget > 0:
            parts.append(f"budget_util={self.rs_bandwidth_utilization[-1]:.4f}")
        print(", ".join(parts))

    def _update_dynamic_prices(self, round_time: float):
        if not self.dynamic_pricing:
            self.rs_price_compute.append(float(self.price_compute))
            self.rs_price_comm.append(float(self.price_comm))
            return

        alpha_t = 0.2
        if self._round_time_ema is None:
            self._round_time_ema = float(round_time)
        else:
            self._round_time_ema = (1.0 - alpha_t) * float(self._round_time_ema) + alpha_t * float(round_time)

        if self.price_target_round_time <= 0.0:
            if len(self._warmup_round_times) < self.price_warmup_rounds:
                self._warmup_round_times.append(float(round_time))
                self.rs_price_compute.append(float(self.price_compute))
                self.rs_price_comm.append(float(self.price_comm))
                return
            target_time = float(np.mean(self._warmup_round_times)) * 0.95
        else:
            target_time = float(self.price_target_round_time)

        acc_delta = 0.0
        if len(self.rs_test_acc_ema) >= 2:
            acc_delta = float(self.rs_test_acc_ema[-1] - self.rs_test_acc_ema[-2])

        mean_stale = float(self.rs_staleness_mean[-1]) if len(self.rs_staleness_mean) > 0 else 0.0
        over_time = float(round_time) > (1.0 + self.price_time_tolerance) * max(target_time, 1e-8)
        acc_drop = acc_delta < -abs(self.price_acc_drop_th)
        acc_gain = acc_delta > abs(self.price_acc_gain_th)
        low_upload_success = bool(len(self.rs_upload_success_ratio) > 0 and self.rs_upload_success_ratio[-1] < 0.5)
        no_usable_delta = bool(len(self.rs_mean_delta_norm) > 0 and self.rs_mean_delta_norm[-1] <= max(self.delta_min_norm * 10.0, 1e-8))
        high_uncertainty = bool(len(self.rs_mean_pred_entropy) > 0 and self.rs_mean_pred_entropy[-1] > self.price_entropy_relax_th)
        using_failover = bool(len(self.rs_failover_used_count) > 0 and self.rs_failover_used_count[-1] > 0)

        if over_time and (not acc_drop) and (not low_upload_success) and (not no_usable_delta):
            self.price_compute = min(self.price_compute_max, self.price_compute + self.price_step)
            self.price_comm = min(self.price_comm_max, self.price_comm + 0.5 * self.price_step)
        elif acc_drop or low_upload_success or no_usable_delta or high_uncertainty:
            self.price_compute = max(self.price_compute_min, self.price_compute - self.price_step)
            self.price_comm = max(self.price_comm_min, self.price_comm - 0.5 * self.price_step)
        elif (not over_time) and acc_gain:
            self.price_comm = max(self.price_comm_min, self.price_comm - 0.25 * self.price_step)

        if mean_stale > self.price_staleness_target or using_failover:
            self.price_comm = max(self.price_comm_min, self.price_comm - 0.5 * self.price_step)

        self.price_compute = float(np.clip(self.price_compute, self.price_compute_min, self.price_compute_max))
        self.price_comm = float(np.clip(self.price_comm, self.price_comm_min, self.price_comm_max))
        self.rs_price_compute.append(float(self.price_compute))
        self.rs_price_comm.append(float(self.price_comm))

    def _reward_reputation(self, client_id: int, good: bool, mild: bool = False):
        rep_decay = float(getattr(self.args, "rep_decay", 0.98))
        rep_bonus = float(getattr(self.args, "rep_bonus", 0.02))
        rep_penalty = float(getattr(self.args, "rep_penalty", 0.02))

        self.reputation[client_id] *= rep_decay
        if good:
            self.reputation[client_id] += rep_bonus
        else:
            self.reputation[client_id] -= rep_penalty * (0.5 if mild else 1.0)
        self.reputation[client_id] = self.reputation[client_id].clamp(0.1, 2.0)

    def _select_auction_winners(self, active_clients: List[object]) -> List[object]:
        if len(active_clients) == 0:
            return []
        if self.current_round < self.warmup_rounds:
            return []

        usable_clients = []
        for client in active_clients:
            is_contract = client.id in self.contract_clients
            if (not is_contract) and str(getattr(client, "last_train_mode", "full")) != "full":
                continue
            if float(getattr(client, "delta_norm", 0.0)) < self.delta_min_norm:
                continue
            usable_clients.append(client)
        if len(usable_clients) == 0:
            return []

        mean_drift = float(np.mean([float(getattr(c, "proto_drift", 0.0)) for c in usable_clients]))

        scored: List[Tuple[float, float, object]] = []
        for client in usable_clients:
            is_contract = client.id in self.contract_clients
            delta_size = float(getattr(client, "delta_size", 1.0))
            delta_mb = max(delta_size / (1024.0 * 1024.0), 1e-6)
            rep = float(self.reputation[client.id].item())
            stale = int(self.staleness[client.id].item())
            drift = float(getattr(client, "proto_drift", 0.0))
            tail_score = float(getattr(client, "tail_score", 0.0))
            train_cost = float(getattr(client, "train_cost_ratio", getattr(client, "train_cost", 1.0)))
            credibility = float(getattr(client, "credibility_score", 0.0))
            quality = float(getattr(client, "quality_score", 0.0))
            risk = float(getattr(client, "risk_score", 0.0))
            pred_entropy = float(getattr(client, "pred_entropy", 0.0))

            if self.max_staleness_upload > 0 and stale > self.max_staleness_upload:
                continue

            mf_penalty = abs(drift - mean_drift)
            stale_bonus = self.stale_score_alpha * min(stale, self.stale_score_cap) if stale >= self.stale_priority_th else 0.0
            entropy_norm = float(np.clip(pred_entropy / max(np.log(max(self.num_classes, 2)), 1e-8), 0.0, 1.0))
            utility = (
                1.20 * credibility
                + 0.40 * quality
                + 0.20 * tail_score
                + 0.15 * entropy_norm
                + 0.10 * rep
                + 0.03 * float(stale)
                + stale_bonus
                + (self.contract_bonus if is_contract else 0.0)
                - 0.25 * self.price_compute * max(train_cost - 1.0, 0.0)
                - 1.00 * self.price_comm * delta_mb
                - 0.10 * mf_penalty
                - 0.10 * risk
            )
            if utility <= 0.0 and (not is_contract):
                continue

            score = utility / delta_mb + (1e3 if is_contract else 0.0)
            scored.append((score, utility, client))

        if len(scored) == 0:
            return []

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        contract_scored = [item for item in scored if item[2].id in self.contract_clients]

        if self.bandwidth_budget > 0:
            winners = []
            used = 0
            for _, _, client in contract_scored:
                sz = int(getattr(client, "delta_size", 0))
                if used + sz <= self.bandwidth_budget:
                    winners.append(client)
                    used += sz
            for _, _, client in scored:
                if client in winners:
                    continue
                sz = int(getattr(client, "delta_size", 0))
                if used + sz <= self.bandwidth_budget:
                    winners.append(client)
                    used += sz
            if len(winners) == 0:
                winners = [scored[0][2]]
            return winners

        k = max(1, int(np.ceil(self.auction_winners_frac * len(usable_clients))))
        k = max(k, len(contract_scored))
        return [client for _, _, client in scored[:k]]

    def _aggregate_base_deltas(self):
        if len(self.uploaded_deltas) == 0:
            return

        base_params = list(self.global_model.base.parameters())
        agg = [torch.zeros_like(p.data) for p in base_params]

        rep_weight = []
        for i, cid in enumerate(self.uploaded_ids):
            stale = float(self.staleness[int(cid)].item()) if int(cid) < len(self.staleness) else 0.0
            stale_decay = float(np.exp(-self.staleness_decay_gamma * max(stale, 0.0)))
            weight_i = float(self.uploaded_weights[i]) * float(self.reputation[int(cid)].item()) * stale_decay
            rep_weight.append(weight_i)
        rep_weight = np.array(rep_weight, dtype=np.float64)
        rep_sum = float(np.sum(rep_weight))
        if rep_sum <= 0.0:
            rep_weight = np.array(self.uploaded_weights, dtype=np.float64)
            rep_sum = float(np.sum(rep_weight)) + 1e-12
        rep_weight = (rep_weight / rep_sum).tolist()

        for w, delta_list in zip(rep_weight, self.uploaded_deltas):
            for j, d in enumerate(delta_list):
                agg[j] += d.to(self.device) * float(w)

        trim_ratio = float(max(0.0, min(0.45, self.delta_trim_ratio)))
        n_up = len(self.uploaded_deltas)
        if n_up >= 5 and trim_ratio > 0.0:
            k = int(np.floor(n_up * trim_ratio))
            if 2 * k < n_up:
                robust = []
                for j in range(len(base_params)):
                    stack_j = torch.stack([dl[j].to(self.device) for dl in self.uploaded_deltas], dim=0)
                    flat = stack_j.view(n_up, -1)
                    sorted_flat, _ = torch.sort(flat, dim=0)
                    trimmed = sorted_flat[k : n_up - k, :]
                    mean_trim = torch.mean(trimmed, dim=0).view_as(base_params[j].data)
                    robust.append(mean_trim)
                rho = 0.5
                agg = [(1.0 - rho) * a + rho * r for a, r in zip(agg, robust)]

        clip = float(self.delta_clip_norm)
        if clip > 0.0:
            total_sq = 0.0
            for d in agg:
                total_sq += float(torch.sum(d * d).item())
            total_norm = float(np.sqrt(max(total_sq, 0.0)))
            if total_norm > clip:
                scale = clip / (total_norm + 1e-12)
                agg = [d * scale for d in agg]

        lr = float(max(0.0, self.delta_server_lr))
        for p, d in zip(base_params, agg):
            p.data += d * lr

    def train_head(self):
        if len(getattr(self, "uploaded_protos", [])) == 0:
            return
        ys = torch.stack([y for _, y in self.uploaded_protos]).view(-1).to(self.device)
        self.head_criterion.update_count(ys)
        proto_loader = DataLoader(self.uploaded_protos, self.batch_size, drop_last=False, shuffle=True)
        epochs = max(1, self.head_train_epochs)
        for _ in range(epochs):
            for p, y in proto_loader:
                p = p.to(self.device, dtype=torch.float32)
                y = y.to(self.device)
                out = self.head(p)
                loss = self.head_criterion(out, y)
                self.opt_h.zero_grad()
                loss.backward()
                self.opt_h.step()

    def train_metrics_personalized(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics_personalized()
            num_samples.append(ns)
            losses.append(cl * 1.0)
        ids = [c.id for c in self.clients]
        return ids, num_samples, losses

    def test_metrics_personalized(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics_personalized()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        return ids, num_samples, tot_correct, tot_auc

    def evaluate_personalized(self, acc=None, loss=None):
        eval_progress_log = bool(getattr(self.args, "eval_progress_log", True))

        if eval_progress_log:
            print(f"[Eval] Start train-metric pass over {len(self.clients)} clients")
        stats_train = self.train_metrics_personalized()
        train_loss = sum(stats_train[2]) * 1.0 / max(sum(stats_train[1]), 1)

        y_true_all = []
        y_prob_all = []
        calib_true_all = []
        calib_prob_all = []
        use_calib = bool(self.eval_calibration and int(self.current_round) >= self.eval_calibration_start_round)
        total_clients = max(len(self.clients), 1)
        for idx, c in enumerate(self.clients, start=1):
            if eval_progress_log:
                print(f"[Eval] client {idx}/{total_clients}: test detail")
            y_true, y_prob = c.test_metrics_detail()
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)
            if use_calib:
                if eval_progress_log:
                    print(f"[Eval] client {idx}/{total_clients}: calibration detail")
                y_true_cal, y_prob_cal = c.calibration_metrics_detail()
                if y_true_cal.size > 0 and y_prob_cal.size > 0:
                    calib_true_all.append(y_true_cal)
                    calib_prob_all.append(y_prob_cal)

        y_true = np.concatenate(y_true_all, axis=0)
        y_prob = np.concatenate(y_prob_all, axis=0)
        y_prob_cal = y_prob
        calib_temp = 1.0
        calib_samples = 0
        if use_calib and len(calib_true_all) > 0 and len(calib_prob_all) > 0:
            calib_true = np.concatenate(calib_true_all, axis=0)
            calib_prob = np.concatenate(calib_prob_all, axis=0)
            calib_samples = int(calib_true.shape[0])
            calib_temp = _fit_temperature(
                calib_true,
                calib_prob,
                t_min=self.eval_calibration_temp_min,
                t_max=self.eval_calibration_temp_max,
                t_steps=self.eval_calibration_temp_steps,
            )
            y_prob_cal = _temperature_scale_probs(y_prob, calib_temp)

        y_pred = np.argmax(y_prob, axis=1)
        y_true_bin = label_binarize(y_true, classes=np.arange(self.num_classes))

        test_acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        micro_auc = _safe_multiclass_roc_auc(y_true_bin, y_prob, average="micro")
        macro_auc = _safe_multiclass_roc_auc(y_true_bin, y_prob, average="macro")
        macro_prauc = _safe_average_precision(y_true_bin, y_prob, average="macro")
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        avg_conf_raw = float(np.mean(np.max(y_prob, axis=1)))
        ece_raw = _expected_calibration_error(y_true, y_prob, n_bins=15)
        brier_raw = float(np.mean(np.sum((y_prob - y_true_bin) ** 2, axis=1)))
        avg_conf = float(np.mean(np.max(y_prob_cal, axis=1)))
        ece = _expected_calibration_error(y_true, y_prob_cal, n_bins=15)
        brier = float(np.mean(np.sum((y_prob_cal - y_true_bin) ** 2, axis=1)))
        micro_fpr = _micro_false_positive_rate(y_true, y_pred, self.num_classes)

        if acc is None:
            self.rs_test_acc.append(float(test_acc))
        else:
            acc.append(float(test_acc))

        if loss is None:
            self.rs_train_loss.append(float(train_loss))
        else:
            loss.append(float(train_loss))

        self.rs_test_auc.append(float(micro_auc))
        self.rs_test_macro_f1.append(float(macro_f1))
        self.rs_test_weighted_f1.append(float(weighted_f1))
        self.rs_test_macro_recall.append(float(macro_recall))
        self.rs_test_balanced_acc.append(float(balanced_acc))
        self.rs_test_macro_auc.append(float(macro_auc))
        self.rs_test_macro_prauc.append(float(macro_prauc))
        self.rs_test_mcc.append(float(mcc))
        self.rs_test_kappa.append(float(kappa))
        self.rs_test_ece.append(float(ece))
        self.rs_test_brier.append(float(brier))
        self.rs_test_micro_fpr.append(float(micro_fpr))
        self.rs_test_avg_conf.append(float(avg_conf))
        self.rs_test_ece_raw.append(float(ece_raw))
        self.rs_test_brier_raw.append(float(brier_raw))
        self.rs_test_avg_conf_raw.append(float(avg_conf_raw))
        self.rs_eval_calibration_temp.append(float(calib_temp))
        self.rs_eval_calibration_samples.append(float(calib_samples))

        alpha = float(max(0.0, min(1.0, self.eval_acc_ema_alpha)))
        if self._eval_acc_ema is None:
            self._eval_acc_ema = float(test_acc)
        else:
            self._eval_acc_ema = (1.0 - alpha) * float(self._eval_acc_ema) + alpha * float(test_acc)
        self.rs_test_acc_ema.append(float(self._eval_acc_ema))

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Macro-F1: {:.4f}".format(macro_f1))
        print("Weighted-F1: {:.4f}".format(weighted_f1))
        print("Balanced Accuracy: {:.4f}".format(balanced_acc))
        print("MCC: {:.4f}".format(mcc))
        print("Cohen-Kappa: {:.4f}".format(kappa))
        print("Micro AUC: {:.4f}".format(micro_auc))
        print("Macro AUC: {:.4f}".format(macro_auc))
        print("Macro PR-AUC: {:.4f}".format(macro_prauc))
        print("Micro FPR: {:.4f}".format(micro_fpr))
        print("ECE: {:.4f}".format(ece))
        print("Brier Score: {:.4f}".format(brier))
        print("Average Confidence: {:.4f}".format(avg_conf))
        if self.eval_calibration:
            print("Calib Temperature: {:.4f}".format(calib_temp))
            print("Calib Samples: {}".format(int(calib_samples)))
            print("ECE(raw): {:.4f}".format(ece_raw))
            print("Brier(raw): {:.4f}".format(brier_raw))
            print("AvgConf(raw): {:.4f}".format(avg_conf_raw))
        print("EMA Test Accuracy: {:.4f}".format(self._eval_acc_ema))

        eval_metrics = {
            "train_loss": float(train_loss),
            "test_acc": float(test_acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "balanced_acc": float(balanced_acc),
            "macro_auc": float(macro_auc),
            "macro_prauc": float(macro_prauc),
            "mcc": float(mcc),
            "kappa": float(kappa),
            "ece": float(ece),
            "brier": float(brier),
            "avg_conf": float(avg_conf),
            "ece_raw": float(ece_raw),
            "brier_raw": float(brier_raw),
            "avg_conf_raw": float(avg_conf_raw),
            "few_shot_recall": 0.0,
            "few_shot_macro_f1": 0.0,
        }
        self._after_personalized_eval(eval_metrics, y_true, y_prob, y_prob_cal)


def _clone_state_dict(state_dict):
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def _expected_calibration_error(y_true, y_prob, n_bins: int = 15):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    conf = np.max(y_prob, axis=1)
    pred = np.argmax(y_prob, axis=1)
    acc = (pred == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        left = bins[i]
        right = bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= left) & (conf <= right)
        else:
            mask = (conf >= left) & (conf < right)
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(acc[mask]))
        bin_conf = float(np.mean(conf[mask]))
        ece += (np.sum(mask) / max(n, 1)) * abs(bin_acc - bin_conf)
    return float(ece)


def _temperature_scale_probs(y_prob: np.ndarray, temperature: float):
    y_prob = np.asarray(y_prob, dtype=np.float64)
    t = float(max(1e-6, temperature))
    logits = np.log(np.clip(y_prob, 1e-12, 1.0))
    scaled = logits / t
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    denom = np.sum(exp_scaled, axis=1, keepdims=True)
    return exp_scaled / np.clip(denom, 1e-12, None)


def _fit_temperature(y_true, y_prob, t_min: float = 0.6, t_max: float = 3.0, t_steps: int = 25):
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.size == 0 or y_prob.size == 0:
        return 1.0

    t_min = float(max(1e-3, t_min))
    t_max = float(max(t_min, t_max))
    t_steps = int(max(2, t_steps))

    idx = np.arange(y_true.shape[0])
    best_t = 1.0
    best_nll = float("inf")
    for t in np.linspace(t_min, t_max, num=t_steps):
        scaled = _temperature_scale_probs(y_prob, float(t))
        picked = scaled[idx, y_true]
        nll = -float(np.mean(np.log(np.clip(picked, 1e-12, 1.0))))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t


def _micro_false_positive_rate(y_true, y_pred, num_classes: int):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (fp + fn + tp)
    return float(fp.sum() / max(fp.sum() + tn.sum(), 1))


def _safe_multiclass_roc_auc(y_true_bin, y_prob, average: str):
    try:
        return float(roc_auc_score(y_true_bin, y_prob, average=average, multi_class="ovr"))
    except ValueError:
        return 0.0


def _safe_average_precision(y_true_bin, y_prob, average: str):
    try:
        return float(average_precision_score(y_true_bin, y_prob, average=average))
    except ValueError:
        return 0.0
