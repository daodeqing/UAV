from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score

from flcore.clients.clientSGEFIDSUS import clientSGEFIDSUS
from flcore.servers.serverHGFIDSUS import HGFIDSUS


class SGEFIDSUS(HGFIDSUS):
    """Prototype-enhanced SGEFIDSUS server."""

    def __init__(self, args, times):
        super().__init__(args, times)

        self.clients = []
        self.set_clients(clientSGEFIDSUS)

        self.enable_contracts = bool(getattr(args, "sgef_enable_contracts", False))
        self.enable_sticky_sampling = bool(getattr(args, "sgef_enable_sticky_sampling", False))
        self.use_failover = bool(getattr(args, "sgef_use_failover", False))

        self.auction_winners_frac = float(getattr(args, "auction_winners_frac", 0.5))
        self.few_shot_selection_bonus = float(getattr(args, "few_shot_selection_bonus", 0.20))
        self.prototype_server_momentum = float(getattr(args, "prototype_server_momentum", 0.20))
        self.global_few_shot_threshold = float(getattr(args, "global_few_shot_threshold", 100.0))

        self.global_prototypes: Dict[int, torch.Tensor] = {}
        self.global_proto_counts: Dict[int, float] = {}

        self.rs_global_proto_classes = []
        self.rs_global_proto_samples = []
        self.rs_few_shot_class_count = []
        self.rs_client_few_shot_ratio = []
        self.rs_test_few_shot_recall = []
        self.rs_test_few_shot_macro_f1 = []

    def _assign_stackelberg_contracts(self):
        if self.enable_contracts:
            return super()._assign_stackelberg_contracts()
        self.contract_clients = set()

    def _broadcast_round_context(self, round_idx: int):
        super()._broadcast_round_context(round_idx)
        for client in self.clients:
            if hasattr(client, "receive_global_prototypes"):
                client.receive_global_prototypes(self.global_prototypes, self.global_proto_counts)

    def receive_models(self, round_idx: int):
        super().receive_models(round_idx)
        self._aggregate_global_prototypes(self.selected_clients)

    def _aggregate_global_prototypes(self, clients: List[object]):
        weighted_sum = defaultdict(lambda: None)
        weight_sum = defaultdict(float)
        support_sum = defaultdict(float)
        few_shot_scores = []

        for client in clients:
            few_shot_scores.append(float(getattr(client, "few_shot_score", 0.0)))
            local_protos = getattr(client, "local_proto_memory", {})
            local_counts = getattr(client, "local_proto_counts", {})
            if len(local_protos) == 0:
                continue

            rep = float(self.reputation[client.id].item()) if client.id < len(self.reputation) else 1.0
            stale = float(self.staleness[client.id].item()) if client.id < len(self.staleness) else 0.0
            client_weight = rep * float(np.exp(-self.staleness_decay_gamma * max(stale, 0.0)))

            for label, proto in local_protos.items():
                label_int = int(label)
                count = float(local_counts.get(label_int, 0.0))
                if count <= 0:
                    continue

                support_sum[label_int] += count
                proto_cpu = proto.detach().cpu().float()
                w = client_weight * max(count, 1.0)

                if weighted_sum[label_int] is None:
                    weighted_sum[label_int] = proto_cpu * w
                else:
                    weighted_sum[label_int] += proto_cpu * w
                weight_sum[label_int] += w

        new_global = {
            int(label): proto.detach().cpu().float().clone()
            for label, proto in self.global_prototypes.items()
        }
        new_counts = {int(label): float(count) for label, count in self.global_proto_counts.items()}

        momentum = float(np.clip(self.prototype_server_momentum, 0.0, 0.99))
        for label, proto_sum in weighted_sum.items():
            avg_proto = proto_sum / max(weight_sum[label], 1e-12)
            if label in new_global:
                avg_proto = momentum * new_global[label].float() + (1.0 - momentum) * avg_proto
            new_global[label] = avg_proto.detach().cpu()
            new_counts[label] = float(support_sum[label])

        self.global_prototypes = new_global
        self.global_proto_counts = new_counts

        self.rs_global_proto_classes.append(float(len(self.global_prototypes)))
        self.rs_global_proto_samples.append(float(sum(self.global_proto_counts.values())))
        self.rs_few_shot_class_count.append(float(len(self._few_shot_labels())))
        self.rs_client_few_shot_ratio.append(float(np.mean(few_shot_scores)) if len(few_shot_scores) > 0 else 0.0)

    def _few_shot_labels(self):
        labels = []
        for label, count in self.global_proto_counts.items():
            if count > 0 and count <= self.global_few_shot_threshold:
                labels.append(int(label))
        return sorted(labels)

    def _after_personalized_eval(
        self,
        metrics: Dict[str, float],
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_prob_cal=None,
    ):
        few_shot_labels = self._few_shot_labels()
        few_shot_recall = 0.0
        few_shot_macro_f1 = 0.0

        if len(few_shot_labels) > 0:
            y_pred = np.argmax(y_prob, axis=1)

            few_mask = np.isin(y_true, few_shot_labels)
            if np.any(few_mask):
                few_true = y_true[few_mask]
                few_pred = y_pred[few_mask]
                few_shot_recall = float(
                    recall_score(
                        few_true,
                        few_pred,
                        labels=few_shot_labels,
                        average="macro",
                        zero_division=0,
                    )
                )
                few_shot_macro_f1 = float(
                    f1_score(
                        few_true,
                        few_pred,
                        labels=few_shot_labels,
                        average="macro",
                        zero_division=0,
                    )
                )

        self.rs_test_few_shot_recall.append(float(few_shot_recall))
        self.rs_test_few_shot_macro_f1.append(float(few_shot_macro_f1))

        print("Few-shot Recall: {:.4f}".format(few_shot_recall))
        print("Few-shot Macro-F1: {:.4f}".format(few_shot_macro_f1))

        metrics = dict(metrics)
        metrics["few_shot_recall"] = float(few_shot_recall)
        metrics["few_shot_macro_f1"] = float(few_shot_macro_f1)
        super()._after_personalized_eval(metrics, y_true, y_prob, y_prob_cal)

    def _select_auction_winners(self, active_clients: List[object]) -> List[object]:
        if len(active_clients) == 0:
            return []
        if self.current_round < self.warmup_rounds:
            return []

        scored: List[Tuple[float, float, object]] = []
        for client in active_clients:
            delta_norm = float(getattr(client, "delta_norm", 0.0))
            if delta_norm < self.delta_min_norm:
                continue

            delta_size = float(max(int(getattr(client, "delta_size", 0)), 1))
            delta_mb = delta_size / (1024.0 * 1024.0)
            utility = float(getattr(client, "bid_value", 0.0))
            utility += self.few_shot_selection_bonus * float(getattr(client, "few_shot_score", 0.0))
            if utility <= 0.0:
                continue

            score = utility / max(delta_mb, 1e-6)
            scored.append((score, utility, client))

        if len(scored) == 0:
            return []

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

        max_winners = max(1, int(np.ceil(self.auction_winners_frac * len(scored))))
        winners = []
        used_bytes = 0

        for _, _, client in scored:
            if len(winners) >= max_winners:
                break
            size_bytes = int(getattr(client, "delta_size", 0))
            if self.bandwidth_budget > 0:
                if used_bytes + size_bytes <= self.bandwidth_budget:
                    winners.append(client)
                    used_bytes += size_bytes
            else:
                winners.append(client)

        if len(winners) == 0 and len(scored) > 0:
            winners = [scored[0][2]]

        return winners

    def _print_round_resource_metrics(self):
        if len(self.rs_total_upload_bytes) > 0:
            parts = [
                "[Round Metrics][SGEFIDSUS]",
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

        if len(self.rs_global_proto_classes) == 0:
            return
        print(
            "[Proto Metrics][SGEFIDSUS] "
            f"global_proto_classes={int(self.rs_global_proto_classes[-1])}, "
            f"global_proto_samples={int(self.rs_global_proto_samples[-1])}, "
            f"few_shot_classes={int(self.rs_few_shot_class_count[-1])}, "
            f"client_few_shot_ratio={self.rs_client_few_shot_ratio[-1]:.4f}"
        )


PSGEFIDSUS = SGEFIDSUS
