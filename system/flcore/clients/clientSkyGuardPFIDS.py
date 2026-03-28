import time
from collections import defaultdict, deque
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from flcore.clients.clientHGFIDSUS import _mmd_fuse_protos, _proto_mean
from flcore.clients.clientSGEFIDSUS import clientSGEFIDSUS


class clientSkyGuardPFIDS(clientSGEFIDSUS):
    """UAV-aware secure personalized FIDS client with cluster adapters."""

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.skyguard_lambda_f = float(getattr(args, "skyguard_lambda_f", 0.35))
        self.skyguard_lambda_p = float(getattr(args, "skyguard_lambda_p", 0.20))
        self.skyguard_focal_gamma = float(getattr(args, "skyguard_focal_gamma", 2.0))
        self.skyguard_entropy_drift_th = float(getattr(args, "skyguard_entropy_drift_th", 0.70))
        self.skyguard_proto_drift_th = float(getattr(args, "skyguard_proto_drift_th", 0.08))
        self.skyguard_personalization_boost = float(getattr(args, "skyguard_personalization_boost", 0.25))

        self.w_gain = float(getattr(args, "skyguard_w_gain", getattr(args, "skyguard_w_shapley", 0.60)))
        self.w_grad = float(getattr(args, "skyguard_w_grad", 0.30))
        self.w_quality = float(getattr(args, "skyguard_w_quality", 0.20))
        self.w_resource = float(getattr(args, "skyguard_w_resource", 0.35))
        self.w_attack = float(getattr(args, "skyguard_w_attack", 0.45))

        self.cluster_id = -1
        self.participation_prob = 1.0
        self.global_drift_alert = False
        self.global_drift_pressure = 0.0
        self.uav_profile = {}
        self.global_delta_reference = None
        self.online_score = 1.0

        self.local_gain_proxy = 0.0
        self.grad_similarity_score = 0.0
        self.data_quality_score = 0.0
        self.resource_score = 0.0
        self.attack_score = 0.0
        self.local_strategy_score = 0.0
        self.utility_uncertainty = 0.0
        self.drift_alert = 0.0
        self.drift_alert_score = 0.0
        self.adapter_drift = 0.0
        self.mode_strategy_payoffs = {}
        self.validation_loss_spike = 0.0
        self.delta_norm_zscore = 0.0
        self.update_cosine_deviation = 0.0
        self.mode_compute_cost = 1.0
        self.reported_comm_cost = 0.0
        self.reported_cost = 0.0

        self._entropy_hist = deque(maxlen=5)
        self._drift_hist = deque(maxlen=5)
        self._utility_hist = deque(maxlen=5)
        self._delta_norm_hist = deque(maxlen=5)
        self._received_cluster_state = None

    def receive_cluster_adapter(self, state_dict: Dict[str, torch.Tensor], cluster_id: int):
        self.cluster_id = int(cluster_id)
        self._received_cluster_state = None
        if state_dict is None or not hasattr(self.model, "adapter"):
            return

        adapter_param = next(self.model.adapter.parameters(), None)
        adapter_device = adapter_param.device if adapter_param is not None else next(self.model.parameters()).device
        clean_state = {
            k: v.detach().clone().to(device=adapter_device)
            for k, v in state_dict.items()
        }
        self.model.adapter.load_state_dict(clean_state, strict=True)
        self.model_per.adapter.load_state_dict(clean_state, strict=True)
        self._received_cluster_state = {
            k: v.detach().cpu().clone()
            for k, v in clean_state.items()
        }

    def receive_uav_context(
        self,
        profile: Dict[str, float],
        cluster_id: int,
        participation_prob: float,
        global_delta_reference,
        drift_alert: bool,
        drift_pressure: float = 0.0,
    ):
        self.cluster_id = int(cluster_id)
        self.uav_profile = dict(profile) if isinstance(profile, dict) else {}
        self.participation_prob = float(participation_prob)
        self.global_drift_alert = bool(drift_alert)
        self.global_drift_pressure = float(drift_pressure)
        self.global_delta_reference = None
        if global_delta_reference is not None:
            self.global_delta_reference = [d.detach().cpu().clone() for d in global_delta_reference]
        self.online_score = float(self.uav_profile.get("availability", self.uav_profile.get("expected_uptime", 1.0)))

    def export_cluster_adapter(self):
        if not hasattr(self.model, "adapter"):
            return {}
        return {k: v.detach().cpu().clone() for k, v in self.model.adapter.state_dict().items()}

    def _forward_parts(self, eval_model, x):
        rep = eval_model.base(x)
        adapted = eval_model.adapter(rep) if hasattr(eval_model, "adapter") else rep
        logits = eval_model.head(adapted)
        return rep, adapted, logits

    def _forward_with_prototypes(self, eval_model, x):
        _, rep, head_logits = self._forward_parts(eval_model, x)
        proto_bank = self._build_prototype_bank(device=rep.device, dtype=rep.dtype)
        if len(proto_bank) == 0 or self.prototype_logit_weight <= 0.0:
            return head_logits
        proto_logits, proto_mask = self._prototype_logits(rep, proto_bank)
        return head_logits + self.prototype_logit_weight * proto_logits * proto_mask

    def _focal_loss(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=1)
        log_prob_true = log_prob.gather(1, y.unsqueeze(1)).squeeze(1)
        prob_true = log_prob_true.exp()
        focal = ((1.0 - prob_true).clamp_min(1e-6) ** self.skyguard_focal_gamma) * (-log_prob_true)
        if sample_weight is not None:
            focal = focal * sample_weight
        return focal.mean()

    def _prototype_consistency_loss(self, rep: torch.Tensor, y: torch.Tensor, batch_protos: Dict[int, torch.Tensor]):
        losses = []
        for label, proto in batch_protos.items():
            mask = y == int(label)
            if not torch.any(mask):
                continue
            proto_vec = proto.to(device=rep.device, dtype=rep.dtype)
            losses.append(torch.mean((rep[mask] - proto_vec.unsqueeze(0)) ** 2))
        if len(losses) == 0:
            return torch.tensor(0.0, device=rep.device, dtype=rep.dtype)
        return torch.stack(losses).mean()

    def _sync_personalized_from_global(self):
        a = float(max(0.0, min(1.0, self.personalized_sync_alpha)))
        if self.global_drift_alert or self.drift_alert > 0.0:
            a = min(0.95, a + self.skyguard_personalization_boost)
        elif self.online_score < 0.45:
            a = min(0.90, a + 0.10)
        for p_per, p_g in zip(self.model_per.base.parameters(), self.model.base.parameters()):
            p_per.data = a * p_per.data + (1.0 - a) * p_g.data
        if hasattr(self.model_per, "adapter") and hasattr(self.model, "adapter"):
            for p_per, p_g in zip(self.model_per.adapter.parameters(), self.model.adapter.parameters()):
                p_per.data = a * p_per.data + (1.0 - a) * p_g.data

    def _apply_train_mode(self, mode: str):
        for p in self.model.head.parameters():
            p.requires_grad = True
        for p in self.model_per.head.parameters():
            p.requires_grad = True

        if hasattr(self.model, "adapter"):
            for p in self.model.adapter.parameters():
                p.requires_grad = True
        if hasattr(self.model_per, "adapter"):
            for p in self.model_per.adapter.parameters():
                p.requires_grad = True

        train_global_base = mode == "full"
        for p in self.model.base.parameters():
            p.requires_grad = bool(train_global_base)

        # Personalized branch stays lightweight: only adapter/head are updated.
        for p in self.model_per.base.parameters():
            p.requires_grad = False

    def _estimate_mode_gain(self, mode: str) -> float:
        drift = float(np.clip(self.drift_alert_score, 0.0, 1.5))
        few = float(np.clip(self.few_shot_ratio, 0.0, 1.0))
        disagree = float(np.clip(1.0 - self.grad_similarity_score, 0.0, 1.0))
        conf = float(np.clip(self.val_confidence, 0.0, 1.0))

        if mode == "full":
            return float(0.45 * drift + 0.30 * few + 0.25 * disagree)
        if mode == "freeze_base":
            return float(0.35 * drift + 0.20 * few + 0.15 * conf)
        return float(0.20 * conf - 0.10 * drift)

    def _estimate_mode_cost(self, mode: str) -> float:
        compute_capacity = float(np.clip(self.uav_profile.get("compute_capacity", 0.5), 1e-3, 1.0))
        link_quality = float(np.clip(self.uav_profile.get("link_quality", 0.5), 1e-3, 1.0))
        mode_comp = {"full": 1.0, "freeze_base": 0.45, "head_only": 0.15}[mode]
        mode_comm = {"full": 1.0, "freeze_base": 0.5, "head_only": 0.15}[mode]
        risk_signal = max(
            float(getattr(self, "risk_score", 0.0)),
            float(np.clip(self.uav_profile.get("attack_exposure", 0.0), 0.0, 2.0)),
        )

        comp_cost = self.server_price_compute * mode_comp / compute_capacity
        comm_cost = self.server_price_comm * mode_comm / link_quality
        risk_cost = 0.15 * risk_signal
        return float(comp_cost + comm_cost + risk_cost)

    def _decide_training_mode(self) -> str:
        candidate_modes = ["full", "freeze_base", "head_only"]
        payoffs = {
            mode: float(self._estimate_mode_gain(mode) - self._estimate_mode_cost(mode))
            for mode in candidate_modes
        }
        self.mode_strategy_payoffs = dict(payoffs)
        return str(max(payoffs.items(), key=lambda item: item[1])[0])

    def _effective_delta_topk(self) -> float:
        topk_ratio = super()._effective_delta_topk()
        if self.last_train_mode == "full" or float(self.few_shot_ratio) >= 0.05:
            topk_ratio = max(topk_ratio, 0.40)
        return float(np.clip(topk_ratio, 1e-6, 1.0))

    def _delta_similarity(self, delta_a, delta_b):
        if delta_a is None or delta_b is None or len(delta_a) == 0 or len(delta_b) == 0:
            return 0.0
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for da, db in zip(delta_a, delta_b):
            da_f = da.detach().float().view(-1).cpu()
            db_f = db.detach().float().view(-1).cpu()
            dot += float(torch.dot(da_f, db_f).item())
            norm_a += float(torch.dot(da_f, da_f).item())
            norm_b += float(torch.dot(db_f, db_f).item())
        if norm_a <= 0.0 or norm_b <= 0.0:
            return 0.0
        cos = dot / (np.sqrt(norm_a) * np.sqrt(norm_b) + 1e-12)
        return float(np.clip(0.5 * (cos + 1.0), 0.0, 1.0))

    def _refresh_drift_alert(self):
        entropy_norm = self._normalize_entropy(float(self.pred_entropy))
        drift_norm = float(np.clip(float(self.proto_drift) / max(self.skyguard_proto_drift_th, 1e-8), 0.0, 1.5))
        ent_hist = [self._normalize_entropy(float(v)) for v in self._entropy_hist]
        drift_hist = [
            float(np.clip(float(v) / max(self.skyguard_proto_drift_th, 1e-8), 0.0, 1.5))
            for v in self._drift_hist
        ]
        ent_mean = float(np.mean(ent_hist)) if len(ent_hist) > 0 else entropy_norm
        drift_mean = float(np.mean(drift_hist)) if len(drift_hist) > 0 else drift_norm
        score = 0.50 * ent_mean + 0.30 * drift_mean + 0.20 * float(np.clip(self.global_drift_pressure, 0.0, 1.5))
        if self.global_drift_alert:
            score += 0.05
        if self.online_score < 0.45:
            score += 0.05
        self.drift_alert_score = float(score)

        trigger_th = 0.95 if self.drift_alert <= 0.0 else 0.82
        local_alert = (
            (entropy_norm >= self.skyguard_entropy_drift_th and float(np.clip(self.global_drift_pressure, 0.0, 1.5)) >= 0.85)
            or (float(self.proto_drift) >= 1.05 * self.skyguard_proto_drift_th)
        )
        alert = local_alert or (score >= trigger_th)
        self.drift_alert = 1.0 if alert else 0.0

    def train(self):
        trainloader, val_loader = self.load_train_data()
        start_time = time.time()

        self.aggregate_parameters(val_loader)

        self.clone_model(self.model, self.old_model)
        shared_ref = [p.detach().clone() for p in self.model.parameters()]
        prev_loss = float(self._recalculate_loss(self.model, val_loader, max_batches=self.bid_eval_batches))
        base_before = [p.detach().clone() for p in self.model.base.parameters()]

        self._train_class_count = self._get_train_class_count().to(self.device)
        local_present = self._train_class_count > 0
        if torch.any(local_present):
            few_shot_present = torch.logical_and(local_present, self._train_class_count <= self.few_shot_threshold)
            self.few_shot_ratio = float(
                few_shot_present.float().sum().item() / max(local_present.float().sum().item(), 1.0)
            )
        else:
            self.few_shot_ratio = 0.0

        self._refresh_drift_alert()
        mode = self._decide_training_mode()
        self.last_train_mode = str(mode)

        self.model.train()
        self.model_per.train()
        self._apply_train_mode(mode)
        self._sync_personalized_from_global()

        protos_g = defaultdict(list)
        protos_per = defaultdict(list)
        cls_cnt = torch.zeros(self.num_classes, device=self.device)
        few_shot_seen = 0.0
        total_seen = 0.0

        for _ in range(self.local_epochs):
            for x, y in trainloader:
                x, y = self._to_device(x, y)
                cls_cnt += torch.bincount(y, minlength=self.num_classes)

                _, rep_g, logits_g = self._forward_parts(self.model, x)
                batch_protos_g, batch_counts_g = self._compute_batch_prototypes(rep_g, y)
                proto_loss_g, combined_logits_g, few_shot_mask_g = self._prototype_enhanced_loss(
                    logits_g,
                    rep_g,
                    y,
                    batch_protos_g,
                    batch_counts_g,
                )
                sample_weight_g, _, _ = self._few_shot_factors(y)
                focal_loss_g = self._focal_loss(combined_logits_g, y, sample_weight=sample_weight_g)
                proto_consistency_g = self._prototype_consistency_loss(rep_g, y, batch_protos_g)
                loss_g = proto_loss_g + self.skyguard_lambda_f * focal_loss_g + self.skyguard_lambda_p * proto_consistency_g

                self.optimizer.zero_grad()
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                _, rep_per, logits_per = self._forward_parts(self.model_per, x)
                batch_protos_per, batch_counts_per = self._compute_batch_prototypes(rep_per, y)
                proto_loss_per, combined_logits_per, few_shot_mask_per = self._prototype_enhanced_loss(
                    logits_per,
                    rep_per,
                    y,
                    batch_protos_per,
                    batch_counts_per,
                )
                sample_weight_per, _, _ = self._few_shot_factors(y)
                focal_loss_per = self._focal_loss(combined_logits_per, y, sample_weight=sample_weight_per)
                proto_consistency_per = self._prototype_consistency_loss(rep_per, y, batch_protos_per)
                loss_per = (
                    proto_loss_per
                    + self.skyguard_lambda_f * focal_loss_per
                    + self.skyguard_lambda_p * proto_consistency_per
                )

                self.optimizer_per.zero_grad()
                loss_per.backward()
                torch.nn.utils.clip_grad_norm_(self.model_per.parameters(), self.grad_clip_norm)
                self.optimizer_per.step(shared_ref, self.device)

                few_shot_seen += float(few_shot_mask_g.float().sum().item() + few_shot_mask_per.float().sum().item())
                total_seen += float(2 * y.numel())

                for idx, yy in enumerate(y):
                    label = int(yy.item())
                    protos_g[label].append(rep_g[idx, :].detach().data)
                    protos_per[label].append(rep_per[idx, :].detach().data)

        protos_g = _proto_mean(protos_g)
        protos_per = _proto_mean(protos_per)

        self.protos_g = protos_g
        self.protos_per = protos_per
        fused_protos = _mmd_fuse_protos(protos_g, protos_per)
        self._update_local_proto_memory(fused_protos, cls_cnt.detach().cpu())
        self.protos = fused_protos

        drift_vals = []
        for label, proto in self.protos.items():
            if label in self.prev_protos:
                cur = F.normalize(proto, dim=0).unsqueeze(0)
                old = F.normalize(self.prev_protos[label], dim=0).unsqueeze(0)
                drift_vals.append(1.0 - F.cosine_similarity(cur, old).item())
        self.proto_drift = float(np.mean(drift_vals)) if len(drift_vals) > 0 else 0.0
        self.prev_protos = {k: v.detach().clone() for k, v in self.protos.items()}

        if self.proto_fp16:
            self.protos = {k: v.detach().cpu().to(torch.float16) for k, v in self.protos.items()}
        else:
            self.protos = {k: v.detach().cpu() for k, v in self.protos.items()}

        self.few_shot_score = float(few_shot_seen / max(total_seen, 1.0))
        self.pred_entropy, self.val_confidence = self._validation_uncertainty(
            self.model, val_loader, max_batches=self.bid_eval_batches
        )
        self._entropy_hist.append(float(self.pred_entropy))
        self._drift_hist.append(float(self.proto_drift))
        self._refresh_drift_alert()

        self.affinity_updated = self._maybe_update_affinity()
        if not self.affinity_updated:
            self.weight_vector = torch.zeros(self.num_clients, device=self.device)
            self._cached_baseline_loss = self._recalculate_loss(
                self.old_model, val_loader, max_batches=self.affinity_eval_batches
            )

        round_train_cost = time.time() - start_time
        self._build_delta_and_bid(prev_loss, base_before, val_loader, cls_cnt, round_train_cost)

        if self._received_cluster_state is not None and hasattr(self.model, "adapter"):
            drift_terms = []
            cur_state = self.model.adapter.state_dict()
            for key, val in cur_state.items():
                old_val = self._received_cluster_state.get(key)
                if old_val is None:
                    continue
                diff = val.detach().cpu().float() - old_val.detach().cpu().float()
                drift_terms.append(float(torch.mean(torch.abs(diff)).item()))
            self.adapter_drift = float(np.mean(drift_terms)) if len(drift_terms) > 0 else 0.0
        else:
            self.adapter_drift = 0.0

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_per.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def _recalculate_loss(self, new_model, val_loader, max_batches: Optional[int] = None):
        if val_loader is None:
            return 0.0
        new_model.eval()
        total = 0.0
        cnt = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x, y = self._to_device(x, y)
                output = self._forward_with_prototypes(new_model, x)
                total += float(
                    F.cross_entropy(
                        output,
                        y,
                        reduction="mean",
                        label_smoothing=float(max(self.label_smoothing, 0.0)),
                    ).item()
                )
                cnt += 1
                if max_batches is not None and (batch_idx + 1) >= max_batches:
                    break
        new_model.train()
        return total / max(cnt, 1)

    def _recalculate_loss_with_peer_base(self, peer_model, val_loader, max_batches: Optional[int] = None):
        if val_loader is None:
            return 0.0
        peer_model.base.eval()
        self.model.head.eval()
        if hasattr(peer_model, "adapter"):
            peer_model.adapter.eval()
        total = 0.0
        cnt = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x, y = self._to_device(x, y)
                rep = peer_model.base(x)
                rep = peer_model.adapter(rep) if hasattr(peer_model, "adapter") else rep
                logits = self.model.head(rep)
                proto_bank = self._build_prototype_bank(device=rep.device, dtype=rep.dtype)
                if len(proto_bank) > 0 and self.prototype_logit_weight > 0.0:
                    proto_logits, proto_mask = self._prototype_logits(rep, proto_bank)
                    logits = logits + self.prototype_logit_weight * proto_logits * proto_mask
                total += float(
                    F.cross_entropy(
                        logits,
                        y,
                        reduction="mean",
                        label_smoothing=float(max(self.label_smoothing, 0.0)),
                    ).item()
                )
                cnt += 1
                if max_batches is not None and (batch_idx + 1) >= max_batches:
                    break
        peer_model.base.train()
        self.model.head.train()
        if hasattr(peer_model, "adapter"):
            peer_model.adapter.train()
        return total / max(cnt, 1)

    def _validation_uncertainty(self, eval_model, val_loader, max_batches: Optional[int] = None):
        if val_loader is None:
            return 0.0, 0.0
        eval_model.eval()
        total_entropy = 0.0
        total_conf = 0.0
        total_n = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x, y = self._to_device(x, y)
                probs = F.softmax(self._forward_with_prototypes(eval_model, x), dim=1)
                entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
                conf = probs.max(dim=1).values
                total_entropy += float(entropy.sum().item())
                total_conf += float(conf.sum().item())
                total_n += int(y.numel())
                if max_batches is not None and (batch_idx + 1) >= max_batches:
                    break
        eval_model.train()
        if total_n <= 0:
            return 0.0, 0.0
        return total_entropy / total_n, total_conf / total_n

    def train_metrics_personalized(self):
        trainloader, _ = self.load_train_data(for_eval=True)
        self.model_per.eval()
        train_num = 0
        losses = 0.0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(trainloader):
                x, y = self._to_device(x, y)
                output = self._forward_with_prototypes(self.model_per, x)
                loss = F.cross_entropy(
                    output,
                    y,
                    reduction="mean",
                    label_smoothing=float(max(self.label_smoothing, 0.0)),
                )
                train_num += y.shape[0]
                losses += float(loss.item()) * y.shape[0]
                if self.eval_train_max_batches > 0 and (batch_idx + 1) >= self.eval_train_max_batches:
                    break
        return losses, train_num

    def test_metrics_personalized(self):
        testloaderfull = self.load_test_data()
        self.model_per.eval()
        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        with torch.no_grad():
            for x, y in testloaderfull:
                x, y = self._to_device(x, y)
                output = self._forward_with_prototypes(self.model_per, x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
                y_prob.append(F.softmax(output, dim=1).detach().cpu().numpy())
                y_true.append(label_binarize(y.detach().cpu().numpy(), classes=np.arange(self.num_classes)))
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        auc = metrics.roc_auc_score(y_true, y_prob, average="micro")
        return test_acc, test_num, auc

    def test_metrics_detail(self):
        testloaderfull = self.load_test_data()
        self.model_per.eval()
        y_true = []
        y_prob = []
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(testloaderfull):
                x, y = self._to_device(x, y)
                output = self._forward_with_prototypes(self.model_per, x)
                y_true.append(y.detach().cpu().numpy())
                y_prob.append(F.softmax(output, dim=1).detach().cpu().numpy())
                if self.eval_test_max_batches > 0 and (batch_idx + 1) >= self.eval_test_max_batches:
                    break
        y_true = np.concatenate(y_true, axis=0)
        y_prob = np.concatenate(y_prob, axis=0)
        return y_true, y_prob

    def calibration_metrics_detail(self):
        val_loader = self.load_val_data()
        self.model_per.eval()
        y_true = []
        y_prob = []
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x, y = self._to_device(x, y)
                output = self._forward_with_prototypes(self.model_per, x)
                y_true.append(y.detach().cpu().numpy())
                y_prob.append(F.softmax(output, dim=1).detach().cpu().numpy())
                if self.eval_calibration_max_batches > 0 and (batch_idx + 1) >= self.eval_calibration_max_batches:
                    break
        if len(y_true) == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0, self.num_classes), dtype=np.float32)
        y_true = np.concatenate(y_true, axis=0)
        y_prob = np.concatenate(y_prob, axis=0)
        return y_true, y_prob

    def _build_delta_and_bid(self, prev_loss, base_before, val_loader, cls_cnt, round_train_cost):
        super()._build_delta_and_bid(prev_loss, base_before, val_loader, cls_cnt, round_train_cost)

        present = cls_cnt > 0
        class_cov = float(present.float().sum().item() / max(self.num_classes, 1))
        if torch.any(present):
            dist = cls_cnt[present].float()
            dist = dist / dist.sum().clamp_min(1.0)
            class_balance = float(
                -(dist * torch.log(dist.clamp_min(1e-12))).sum().item() / max(np.log(max(int(dist.numel()), 2)), 1e-8)
            )
        else:
            class_balance = 0.0

        entropy_norm = self._normalize_entropy(float(self.pred_entropy))
        drift_norm = float(np.clip(self.proto_drift / max(2.0 * self.skyguard_proto_drift_th, 1e-8), 0.0, 1.5))
        rep_term = float(np.clip(self.server_reputation / 2.0, 0.0, 1.0))

        self.local_gain_proxy = float(
            0.60 * self.bid_loss_drop
            + 0.20 * self.few_shot_ratio
            + 0.20 * self.val_confidence
        )
        self.grad_similarity_score = float(self._delta_similarity(self.delta_base, self.global_delta_reference))
        self.update_cosine_deviation = float(np.clip(1.0 - self.grad_similarity_score, 0.0, 1.0))
        self.data_quality_score = float(0.35 * class_cov + 0.35 * class_balance + 0.30 * self.val_confidence)

        compute_capacity = float(self.uav_profile.get("compute_capacity", 0.5))
        link_quality = float(self.uav_profile.get("link_quality", self.uav_profile.get("base_link_quality", 0.5)))
        energy_level = float(self.uav_profile.get("energy_level", self.uav_profile.get("base_energy_level", 0.5)))
        expected_uptime = float(self.uav_profile.get("expected_uptime", self.online_score))
        time_eff = float(1.0 / max(self.train_cost_ratio, 1.0))
        self.resource_score = float(
            np.clip(
                0.25 * compute_capacity + 0.25 * link_quality + 0.25 * energy_level + 0.15 * expected_uptime + 0.10 * time_eff,
                0.0,
                1.0,
            )
        )

        self.validation_loss_spike = float(max(float(self.last_val_loss) - float(prev_loss), 0.0))
        hist_mean = float(np.mean(self._delta_norm_hist)) if len(self._delta_norm_hist) > 0 else float(self.delta_norm)
        hist_std = float(np.std(self._delta_norm_hist)) if len(self._delta_norm_hist) > 1 else 0.0
        self.delta_norm_zscore = float(abs(self.delta_norm - hist_mean) / max(hist_std, 1e-6)) if hist_std > 0 else 0.0
        self._delta_norm_hist.append(float(self.delta_norm))
        loss_spike_norm = float(np.clip(self.validation_loss_spike / max(float(prev_loss), 1e-6), 0.0, 1.5))
        norm_z_norm = float(np.clip(self.delta_norm_zscore / 3.0, 0.0, 1.5))
        support_shift = float(abs(self.global_proto_coverage - class_cov))
        self.attack_score = float(
            np.clip(
                0.25 * entropy_norm
                + 0.20 * drift_norm
                + 0.20 * self.update_cosine_deviation
                + 0.15 * loss_spike_norm
                + 0.10 * norm_z_norm
                + 0.10 * support_shift
                + 0.10 * (1.0 - rep_term),
                0.0,
                2.0,
            )
        )
        self.risk_score = float(self.attack_score)

        self.mode_compute_cost = {"full": 1.0, "freeze_base": 0.45, "head_only": 0.15}[self.last_train_mode]
        self.reported_comm_cost = float(self.delta_size) / (1024.0 * 1024.0)
        self.reported_cost = (
            float(self.server_price_compute) * float(self.mode_compute_cost)
            + float(self.server_price_comm) * float(self.reported_comm_cost)
        )

        strategy_score = (
            self.w_gain * self.local_gain_proxy
            + self.w_grad * self.grad_similarity_score
            + self.w_quality * self.data_quality_score
            + self.w_resource * self.resource_score
            - self.w_attack * self.attack_score
            - self.reported_cost
        )

        self.quality_score = float(0.60 * self.data_quality_score + 0.40 * self.local_gain_proxy)
        self.credibility_score = float(
            (self.quality_score + 1e-6) * (1.0 + 0.25 * self.server_reputation) / (1.0 + self.attack_score)
        )
        self.local_strategy_score = float(strategy_score)
        self.bid_value = float(max(self.local_gain_proxy - self.reported_cost, 0.0))

        self._utility_hist.append(float(self.local_strategy_score))
        util_std = float(np.std(list(self._utility_hist))) if len(self._utility_hist) > 1 else 0.0
        ent_std = float(np.std(list(self._entropy_hist))) if len(self._entropy_hist) > 1 else 0.0
        drift_std = float(np.std(list(self._drift_hist))) if len(self._drift_hist) > 1 else 0.0
        self.utility_uncertainty = float(util_std + 0.5 * ent_std + 0.5 * drift_std)
