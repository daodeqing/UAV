import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from flcore.clients.clientHGFIDSUS import clientHGFIDSUS, _mmd_fuse_protos, _proto_mean


class clientSGEFIDSUS(clientHGFIDSUS):
    """Prototype-enhanced SGEFIDSUS client."""

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.u_gain = float(getattr(args, "u_gain", 1.0))
        self.u_tail = float(getattr(args, "u_tail", 0.25))
        self.u_uncertainty = float(getattr(args, "u_uncertainty", 0.2))
        self.u_rep = float(getattr(args, "u_rep", 0.15))
        self.u_few_shot = float(getattr(args, "u_few_shot", 0.20))

        self.prototype_logit_weight = float(getattr(args, "prototype_logit_weight", 0.60))
        self.prototype_loss_weight = float(getattr(args, "prototype_loss_weight", 0.25))
        self.prototype_temperature = float(getattr(args, "prototype_temperature", 1.0))
        self.prototype_memory_momentum = float(getattr(args, "prototype_memory_momentum", 0.80))
        self.prototype_global_mix = float(getattr(args, "prototype_global_mix", 0.55))
        self.prototype_l2_norm = bool(getattr(args, "prototype_l2_norm", True))

        self.few_shot_threshold = int(getattr(args, "few_shot_threshold", 20))
        self.few_shot_loss_boost = float(getattr(args, "few_shot_loss_boost", 0.50))
        self.few_shot_proto_boost = float(getattr(args, "few_shot_proto_boost", 0.40))
        self.stale_score_alpha = float(getattr(args, "stale_score_alpha", 0.25))
        self.stale_score_cap = int(getattr(args, "stale_score_cap", 30))

        self.local_proto_memory: Dict[int, torch.Tensor] = {}
        self.local_proto_counts: Dict[int, float] = {}
        self.global_proto_memory: Dict[int, torch.Tensor] = {}
        self.global_proto_counts: Dict[int, float] = {}

        self.few_shot_score = 0.0
        self.few_shot_ratio = 0.0
        self.global_proto_coverage = 0.0
        self._train_class_count = None

    def receive_global_prototypes(self, prototypes: Dict[int, torch.Tensor], counts: Dict[int, float]):
        self.global_proto_memory = {
            int(label): proto.detach().cpu().float().clone()
            for label, proto in prototypes.items()
            if proto is not None
        }
        self.global_proto_counts = {int(label): float(counts.get(label, 0.0)) for label in self.global_proto_memory}
        self.global_proto_coverage = float(len(self.global_proto_memory) / max(self.num_classes, 1))

    def _get_train_class_count(self) -> torch.Tensor:
        self._ensure_train_val_cache()
        if self._train_class_count is not None:
            return self._train_class_count.clone()

        if self._cached_train_data is None or len(self._cached_train_data) == 0:
            self._train_class_count = torch.zeros(self.num_classes, dtype=torch.long)
            return self._train_class_count.clone()

        labels = []
        for _, y in self._cached_train_data:
            labels.append(int(y.item()) if torch.is_tensor(y) else int(y))
        y_tensor = torch.tensor(labels, dtype=torch.long)
        self._train_class_count = torch.bincount(y_tensor, minlength=self.num_classes).cpu()
        return self._train_class_count.clone()

    def _compute_batch_prototypes(self, rep: torch.Tensor, y: torch.Tensor) -> Tuple[Dict[int, torch.Tensor], Dict[int, float]]:
        batch_protos: Dict[int, torch.Tensor] = {}
        batch_counts: Dict[int, float] = {}
        for label in torch.unique(y):
            label_int = int(label.item())
            mask = y == label
            batch_protos[label_int] = rep[mask].detach().mean(dim=0)
            batch_counts[label_int] = float(mask.sum().item())
        return batch_protos, batch_counts

    def _build_prototype_bank(
        self,
        batch_protos: Optional[Dict[int, torch.Tensor]] = None,
        batch_counts: Optional[Dict[int, float]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[int, torch.Tensor]:
        device = self.device if device is None else device
        dtype = torch.float32 if dtype is None else dtype

        proto_bank: Dict[int, torch.Tensor] = {}
        for label in range(self.num_classes):
            local_proto = None
            global_proto = None

            if batch_protos is not None and label in batch_protos:
                local_proto = batch_protos[label].to(device=device, dtype=dtype)
            elif label in self.local_proto_memory:
                local_proto = self.local_proto_memory[label].to(device=device, dtype=dtype)

            if label in self.global_proto_memory:
                global_proto = self.global_proto_memory[label].to(device=device, dtype=dtype)

            if local_proto is None and global_proto is None:
                continue

            local_count = float(self.local_proto_counts.get(label, 0.0))
            if batch_counts is not None and label in batch_counts:
                local_count = float(batch_counts[label])
            global_count = float(self.global_proto_counts.get(label, 0.0))

            if local_proto is not None and global_proto is not None:
                global_mix = float(np.clip(self.prototype_global_mix, 0.0, 0.95))
                if local_count <= self.few_shot_threshold:
                    global_mix = float(np.clip(global_mix + self.few_shot_proto_boost, 0.0, 0.98))
                local_weight = max(local_count, 1.0)
                global_weight = max(global_count, 1.0) * max(global_mix, 1e-6)
                proto = (local_proto * local_weight + global_proto * global_weight) / (local_weight + global_weight)
            elif local_proto is not None:
                proto = local_proto
            else:
                proto = global_proto

            if self.prototype_l2_norm:
                proto = F.normalize(proto, dim=0)
            proto_bank[label] = proto

        return proto_bank

    def _prototype_logits(self, rep: torch.Tensor, proto_bank: Dict[int, torch.Tensor]):
        proto_logits = torch.zeros(rep.shape[0], self.num_classes, device=rep.device, dtype=rep.dtype)
        proto_mask = torch.zeros_like(proto_logits)
        if len(proto_bank) == 0:
            return proto_logits, proto_mask

        rep_eval = F.normalize(rep, dim=1) if self.prototype_l2_norm else rep
        temp = max(self.prototype_temperature, 1e-6)

        for label, proto in proto_bank.items():
            proto_vec = proto.to(device=rep.device, dtype=rep.dtype)
            diff = rep_eval - proto_vec.unsqueeze(0)
            dist = torch.sum(diff * diff, dim=1)
            proto_logits[:, int(label)] = -dist / temp
            proto_mask[:, int(label)] = 1.0

        return proto_logits, proto_mask

    def _few_shot_factors(self, y: torch.Tensor):
        if self._train_class_count is None:
            class_count = torch.full((self.num_classes,), self.few_shot_threshold + 1, device=y.device)
        else:
            class_count = self._train_class_count.to(y.device)

        support = class_count[y]
        few_shot_mask = support <= self.few_shot_threshold
        sample_weight = 1.0 + self.few_shot_loss_boost * few_shot_mask.float()
        proto_scale = self.prototype_logit_weight * (1.0 + self.few_shot_proto_boost * few_shot_mask.float())
        return sample_weight, proto_scale, few_shot_mask

    def _classification_loss(self, logits: torch.Tensor, y: torch.Tensor, sample_weight: Optional[torch.Tensor] = None):
        losses = F.cross_entropy(
            logits,
            y,
            reduction="none",
            label_smoothing=float(max(self.label_smoothing, 0.0)),
        )
        if sample_weight is not None:
            losses = losses * sample_weight
        return losses.mean()

    def _prototype_enhanced_loss(
        self,
        head_logits: torch.Tensor,
        rep: torch.Tensor,
        y: torch.Tensor,
        batch_protos: Dict[int, torch.Tensor],
        batch_counts: Dict[int, float],
    ):
        proto_bank = self._build_prototype_bank(
            batch_protos=batch_protos,
            batch_counts=batch_counts,
            device=rep.device,
            dtype=rep.dtype,
        )
        if len(proto_bank) == 0 or self.prototype_logit_weight <= 0.0:
            return self._classification_loss(head_logits, y), head_logits, torch.zeros_like(y, dtype=torch.bool)

        proto_logits, proto_mask = self._prototype_logits(rep, proto_bank)
        sample_weight, proto_scale, few_shot_mask = self._few_shot_factors(y)
        combined_logits = head_logits + proto_scale.unsqueeze(1) * proto_logits * proto_mask

        cls_loss = self._classification_loss(combined_logits, y, sample_weight=sample_weight)
        masked_proto_logits = torch.where(
            proto_mask > 0,
            proto_logits,
            torch.full_like(proto_logits, -1e4),
        )
        proto_loss = self._classification_loss(masked_proto_logits, y, sample_weight=sample_weight)
        total_loss = cls_loss + self.prototype_loss_weight * proto_loss
        return total_loss, combined_logits, few_shot_mask

    def _forward_with_prototypes(self, eval_model, x):
        rep = eval_model.base(x)
        head_logits = eval_model.head(rep)
        proto_bank = self._build_prototype_bank(device=rep.device, dtype=rep.dtype)
        if len(proto_bank) == 0 or self.prototype_logit_weight <= 0.0:
            return head_logits
        proto_logits, proto_mask = self._prototype_logits(rep, proto_bank)
        return head_logits + self.prototype_logit_weight * proto_logits * proto_mask

    def _update_local_proto_memory(self, protos: Dict[int, torch.Tensor], class_count: torch.Tensor):
        momentum = float(np.clip(self.prototype_memory_momentum, 0.0, 0.99))
        for label, proto in protos.items():
            label_int = int(label)
            count = float(class_count[label_int].item()) if label_int < len(class_count) else 0.0
            if count <= 0:
                continue

            proto_cpu = proto.detach().cpu().float()
            if label_int in self.local_proto_memory:
                mixed = momentum * self.local_proto_memory[label_int].float() + (1.0 - momentum) * proto_cpu
                self.local_proto_memory[label_int] = mixed.detach().clone()
            else:
                self.local_proto_memory[label_int] = proto_cpu
            self.local_proto_counts[label_int] = count

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
            self.few_shot_ratio = float(few_shot_present.float().sum().item() / max(local_present.float().sum().item(), 1.0))
        else:
            self.few_shot_ratio = 0.0

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

                rep = self.model.base(x)
                batch_protos_g, batch_counts_g = self._compute_batch_prototypes(rep, y)
                logits = self.model.head(rep)
                loss, _, few_shot_mask = self._prototype_enhanced_loss(logits, rep, y, batch_protos_g, batch_counts_g)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                rep_per = self.model_per.base(x)
                batch_protos_per, batch_counts_per = self._compute_batch_prototypes(rep_per, y)
                logits_per = self.model_per.head(rep_per)
                loss_per, _, few_shot_mask_per = self._prototype_enhanced_loss(
                    logits_per,
                    rep_per,
                    y,
                    batch_protos_per,
                    batch_counts_per,
                )
                self.optimizer_per.zero_grad()
                loss_per.backward()
                torch.nn.utils.clip_grad_norm_(self.model_per.parameters(), self.grad_clip_norm)
                self.optimizer_per.step(shared_ref, self.device)

                few_shot_seen += float(few_shot_mask.float().sum().item() + few_shot_mask_per.float().sum().item())
                total_seen += float(2 * y.numel())

                for idx, yy in enumerate(y):
                    label = int(yy.item())
                    protos_g[label].append(rep[idx, :].detach().data)
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

        self.affinity_updated = self._maybe_update_affinity()
        if not self.affinity_updated:
            self.weight_vector = torch.zeros(self.num_clients, device=self.device)
            self._cached_baseline_loss = self._recalculate_loss(
                self.old_model, val_loader, max_batches=self.affinity_eval_batches
            )

        round_train_cost = time.time() - start_time
        self._build_delta_and_bid(prev_loss, base_before, val_loader, cls_cnt, round_train_cost)

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
        total = 0.0
        cnt = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x, y = self._to_device(x, y)
                rep = peer_model.base(x)
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
            for x, y in trainloader:
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
            for x, y in testloaderfull:
                x, y = self._to_device(x, y)
                output = self._forward_with_prototypes(self.model_per, x)
                y_true.append(y.detach().cpu().numpy())
                y_prob.append(F.softmax(output, dim=1).detach().cpu().numpy())
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
        present = cls_cnt > 0
        if torch.any(present):
            self.tail_score = float((1.0 / (cls_cnt[present] + 1.0).sqrt()).mean().item())
            few_shot_present = torch.logical_and(present, cls_cnt <= self.few_shot_threshold)
            self.few_shot_ratio = float(few_shot_present.float().sum().item() / max(present.float().sum().item(), 1.0))
        else:
            self.tail_score = 0.0
            self.few_shot_ratio = 0.0

        raw_delta = []
        for p_new, p_old in zip(self.model.base.parameters(), base_before):
            raw_delta.append((p_new.detach() - p_old).clone())

        delta_list, sparse_bytes = self._compress_base_delta(raw_delta)
        sq = 0.0
        non_zero = 0
        total_numel = 0
        for d in delta_list:
            sq += float((d * d).sum().item())
            non_zero += int((d != 0).sum().item())
            total_numel += int(d.numel())
        self.delta_base = delta_list
        self.delta_norm = float(np.sqrt(max(sq, 0.0)))

        dense_bytes = int(sum(d.numel() * d.element_size() for d in self.delta_base))
        self.delta_size = int(min(dense_bytes, sparse_bytes))
        self.delta_sparsity = 1.0 - float(non_zero / max(total_numel, 1))

        new_val_loss = (
            float(self._recalculate_loss(self.model, val_loader, max_batches=self.bid_eval_batches))
            if val_loader is not None
            else prev_loss
        )
        loss_drop = float(max(prev_loss - new_val_loss, 0.0))
        self.last_val_loss = new_val_loss

        self.bid_loss_drop = float(loss_drop)
        self.bid_proto_drift = float(self.proto_drift)
        self.bid_staleness_term = float(self.server_staleness)

        self.train_cost = float(round_train_cost)
        if self.train_time_cost["num_rounds"] > 0:
            hist_avg = max(self.train_time_cost["total_cost"] / self.train_time_cost["num_rounds"], 1e-6)
            self.train_cost_ratio = float(round_train_cost / hist_avg)
        else:
            self.train_cost_ratio = 1.0

        comm_cost = float(self.delta_size) / (1024.0 * 1024.0)
        entropy_norm = self._normalize_entropy(float(self.pred_entropy))

        eff_price_compute = float(self.server_price_compute) * (1.0 - float(self.server_contract_discount_compute))
        eff_price_comm = float(self.server_price_comm) * (1.0 - float(self.server_contract_discount_comm))

        stale_bonus = min(int(self.server_staleness), int(self.stale_score_cap)) * float(self.stale_score_alpha)
        mode_compute_factor = 1.0 if self.last_train_mode == "full" else (0.45 if self.last_train_mode == "freeze_base" else 0.15)

        utility = (
            self.u_gain * float(loss_drop)
            + self.u_tail * float(self.tail_score)
            + self.u_uncertainty * float(entropy_norm)
            + self.u_rep * float(self.server_reputation)
            + self.u_few_shot * float(self.few_shot_ratio)
            + float(stale_bonus)
            + float(self.server_contract_bonus)
            - eff_price_compute * mode_compute_factor
            - eff_price_comm * comm_cost
        )

        self.quality_score = float(loss_drop + 0.30 * self.tail_score + 0.25 * self.few_shot_ratio)
        self.risk_score = float(entropy_norm)
        self.credibility_score = float(
            (self.quality_score + 1e-6)
            * (1.0 + 0.2 * float(self.server_reputation))
            / (1.0 + self.risk_score)
        )
        self.delta_quality = float(loss_drop / max(comm_cost, 1e-6)) if comm_cost > 0 else float(loss_drop)
        self.bid_value = float(max(utility, 0.0))
