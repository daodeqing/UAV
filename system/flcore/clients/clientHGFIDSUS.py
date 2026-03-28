import copy
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.utils.data import DataLoader

from flcore.clients.clientbase import Client
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
from utils.data_utils import read_client_data_un


class clientHGFIDSUS(Client):
    """Client for HGFIDSUS.

    This version keeps the original HGF/FIDSUS spirit, but adds:
    - entropy-aware fuzzy mode selection (full / freeze_base / head_only)
    - split-style personalization: freeze_base only freezes the global base
    - cheap validation for affinity / bidding to reduce per-round overhead
    - cached train/val/test datasets to avoid repeated file IO
    - credibility / uncertainty statistics for game-theoretic winner selection
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.num_clients = args.num_clients
        self.seed = int(getattr(args, "seed", 0)) + 10007 * int(id)
        self.loader_num_workers = int(getattr(args, "loader_num_workers", 0))
        self.cache_client_data = bool(getattr(args, "cache_client_data", True))
        self.pin_memory = bool("cuda" in str(self.device))
        self.eval_loader_num_workers = int(getattr(args, "eval_loader_num_workers", 0))
        self.eval_pin_memory = bool(getattr(args, "eval_pin_memory", False))
        self.eval_calibration_max_batches = int(getattr(args, "eval_calibration_max_batches", 0))
        self.eval_train_max_batches = int(getattr(args, "eval_train_max_batches", 0))
        self.eval_test_max_batches = int(getattr(args, "eval_test_max_batches", 0))

        self.old_model = copy.deepcopy(self.model)

        self.received_ids: List[int] = []
        self.received_models: List[torch.nn.Module] = []

        self.weight_vector = torch.zeros(self.num_clients, device=self.device)
        self._cached_baseline_loss = 0.0

        self.val_ratio = float(getattr(args, "val_ratio", 0.1))
        self.mu = args.mu

        # Personalized branch (as in FIDSUS)
        self.model_per = copy.deepcopy(self.model)
        self.optimizer_per = PerturbedGradientDescent(self.model_per.parameters(), lr=self.learning_rate, mu=self.mu)
        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per, gamma=args.learning_rate_decay_gamma
        )

        self.CEloss = nn.CrossEntropyLoss()
        self.head_per = self.model_per.head
        self.opt_h_per = torch.optim.SGD(self.head_per.parameters(), lr=self.learning_rate)

        # Server broadcast context (filled each round)
        self.server_price_compute = 1.0
        self.server_price_comm = 1.0
        self.force_affinity_update = False
        self.round_idx = 0
        self.server_reputation = 1.0
        self.server_staleness = 0
        self.server_mean_field_drift = 0.0
        self.server_contract_mode = ""
        self.server_contract_bonus = 0.0
        self.server_contract_discount_compute = 0.0
        self.server_contract_discount_comm = 0.0

        # Tracking for sparse updates and bidding
        self.prev_protos: Dict[int, torch.Tensor] = {}
        self.proto_drift = 0.0
        self.affinity_updated = True

        self.delta_base = None
        self.delta_norm = 0.0
        self.delta_size = 0
        self.bid_value = 0.0
        self.tail_score = 0.0
        self.train_cost = 0.0
        self.train_cost_ratio = 1.0

        # Expose components for server-side HGF-specific metrics
        self.last_train_mode = "full"
        self.bid_loss_drop = 0.0
        self.bid_proto_drift = 0.0
        self.bid_staleness_term = 0.0
        self.pred_entropy = 0.0
        self.val_confidence = 0.0
        self.credibility_score = 0.0
        self.quality_score = 0.0
        self.risk_score = 0.0
        self.delta_quality = 0.0
        self.delta_sparsity = 0.0
        self._delta_residual = None

        self.last_val_loss = None

        # Hyperparameters
        self.affinity_drift_th = float(getattr(args, "affinity_drift_th", 0.05))
        self.affinity_min_prob = float(getattr(args, "affinity_min_prob", 0.2))
        self.bid_w1 = float(getattr(args, "bid_w1", 1.0))
        self.bid_w2 = float(getattr(args, "bid_w2", 1.0))
        self.bid_w3 = float(getattr(args, "bid_w3", 0.1))

        # Stability knobs
        self.label_smoothing = float(getattr(args, "label_smoothing", 0.05))
        self.grad_clip_norm = float(getattr(args, "grad_clip_norm", 5.0))
        self.peer_mixing = float(getattr(args, "peer_mixing", 0.6))
        self.personalized_sync_alpha = float(getattr(args, "personalized_sync_alpha", 0.3))

        # New knobs from the latest HG update.
        self.affinity_eval_batches = int(getattr(args, "affinity_eval_batches", 2))
        self.bid_eval_batches = int(getattr(args, "bid_eval_batches", 2))
        self.full_train_rounds = int(getattr(args, "full_train_rounds", 20))
        self.freeze_base_price = float(getattr(args, "freeze_base_price", 1.35))
        self.head_only_price = float(getattr(args, "head_only_price", 1.85))
        self.entropy_keep_full_th = float(getattr(args, "entropy_keep_full_th", 0.55))
        self.drift_keep_full_th = float(getattr(args, "drift_keep_full_th", max(self.affinity_drift_th, 0.08)))
        self.stale_force_full_th = int(getattr(args, "stale_force_full_th", 12))
        self.compute_cost_weight = float(getattr(args, "compute_cost_weight", 0.20))
        self.comm_cost_weight = float(getattr(args, "comm_cost_weight", 1.00))
        self.uncertainty_bonus_weight = float(getattr(args, "uncertainty_bonus_weight", 0.20))
        self.enable_delta_compression = bool(getattr(args, "enable_delta_compression", True))
        self.delta_topk = float(getattr(args, "delta_topk", 0.10))
        self.delta_error_feedback = bool(getattr(args, "delta_error_feedback", True))
        self.delta_topk_warmup_rounds = int(getattr(args, "delta_topk_warmup_rounds", 15))
        self.delta_topk_warmup_value = float(getattr(args, "delta_topk_warmup_value", 0.35))
        self.contract_force_dense_delta = bool(getattr(args, "contract_force_dense_delta", True))
        self.delta_tail_boost = float(getattr(args, "delta_tail_boost", 0.20))
        self.delta_cred_boost = float(getattr(args, "delta_cred_boost", 0.20))
        self.delta_adaptive_topk_max = float(getattr(args, "delta_adaptive_topk_max", 0.60))
        self.proto_fp16 = bool(getattr(args, "proto_fp16", True))

        self._cached_train_data = None
        self._cached_val_data = None
        self._cached_test_data = None

        if self.label_smoothing > 0:
            self.loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def _make_loader(self, data, batch_size=None, shuffle=False, is_train=False):
        if batch_size is None:
            batch_size = self.batch_size
        if is_train:
            num_workers = self.loader_num_workers
            pin_memory = self.pin_memory
        else:
            num_workers = self.eval_loader_num_workers
            pin_memory = self.eval_pin_memory
        return DataLoader(
            data,
            batch_size=batch_size,
            drop_last=False,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def _ensure_train_val_cache(self):
        if self._cached_train_data is not None and self._cached_val_data is not None:
            return

        train_data = read_client_data_un(self.dataset, self.id, is_train=True)
        n = len(train_data)
        if n == 0:
            self._cached_train_data = train_data
            self._cached_val_data = train_data
            self.train_samples = 0
            return

        if n < 5 or self.val_ratio <= 0.0:
            self._cached_train_data = train_data
            self._cached_val_data = train_data
            self.train_samples = len(train_data)
            return

        n_val = max(1, int(round(self.val_ratio * n)))
        n_val = min(n - 1, n_val)
        idx = np.arange(n)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(idx)
        val_idx = idx[:n_val].tolist()
        train_idx = idx[n_val:].tolist()
        self._cached_train_data = [train_data[i] for i in train_idx]
        self._cached_val_data = [train_data[i] for i in val_idx]
        self.train_samples = len(self._cached_train_data)

    def load_train_data(self, batch_size=None, for_eval=False):
        self._ensure_train_val_cache()
        return (
            self._make_loader(
                self._cached_train_data,
                batch_size=batch_size,
                shuffle=(not for_eval),
                is_train=(not for_eval),
            ),
            self._make_loader(self._cached_val_data, batch_size=batch_size, shuffle=False, is_train=False),
        )

    def load_test_data(self, batch_size=None):
        if self._cached_test_data is None:
            self._cached_test_data = read_client_data_un(self.dataset, self.id, is_train=False)
        return self._make_loader(self._cached_test_data, batch_size=batch_size, shuffle=False, is_train=False)

    def load_val_data(self, batch_size=None):
        self._ensure_train_val_cache()
        return self._make_loader(self._cached_val_data, batch_size=batch_size, shuffle=False, is_train=False)

    def receive_models(self, ids, models):
        self.received_ids = ids
        self.received_models = models

    def set_parameters(self, head):
        for new_param, old_param in zip(head.parameters(), self.model.head.parameters()):
            old_param.data = new_param.data.clone()
        for new_param, old_param in zip(head.parameters(), self.model_per.head.parameters()):
            old_param.data = new_param.data.clone()

    def set_global_base(self, global_base, alpha=0.05):
        a = float(max(0.0, min(1.0, alpha)))
        if a <= 0.0:
            return
        for lp, gp in zip(self.model.base.parameters(), global_base.parameters()):
            lp.data = (1.0 - a) * lp.data + a * gp.data.clone()
        for lp, gp in zip(self.model_per.base.parameters(), global_base.parameters()):
            lp.data = (1.0 - a) * lp.data + a * gp.data.clone()

    def train(self):
        trainloader, val_loader = self.load_train_data()
        start_time = time.time()

        # Aggregate from received peer models (FIDSUS core) before local updates.
        self.aggregate_parameters(val_loader)

        self.clone_model(self.model, self.old_model)
        shared_ref = [p.detach().clone() for p in self.model.parameters()]
        prev_loss = float(self._recalculate_loss(self.model, val_loader, max_batches=self.bid_eval_batches))
        base_before = [p.detach().clone() for p in self.model.base.parameters()]

        mode = self._decide_training_mode()
        self.last_train_mode = str(mode)

        self.model.train()
        self.model_per.train()
        self._apply_train_mode(mode)
        self._sync_personalized_from_global()

        protos_g = defaultdict(list)
        protos_per = defaultdict(list)
        cls_cnt = torch.zeros(self.num_classes, device=self.device)

        max_local_epochs = self.local_epochs
        for _ in range(max_local_epochs):
            for x, y in trainloader:
                x, y = self._to_device(x, y)
                cls_cnt += torch.bincount(y, minlength=self.num_classes)

                rep = self.model.base(x)
                out = self.model.head(rep)
                loss = self.loss(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                rep_per = self.model_per.base(x)
                out_per = self.model_per.head(rep_per)
                loss_per = self.loss(out_per, y)
                self.optimizer_per.zero_grad()
                loss_per.backward()
                torch.nn.utils.clip_grad_norm_(self.model_per.parameters(), self.grad_clip_norm)
                self.optimizer_per.step(shared_ref, self.device)

                for i, yy in enumerate(y):
                    label = int(yy.item())
                    protos_g[label].append(rep[i, :].detach().data)
                    protos_per[label].append(rep_per[i, :].detach().data)

        protos_g = _proto_mean(protos_g)
        protos_per = _proto_mean(protos_per)

        self.protos_g = protos_g
        self.protos_per = protos_per
        self.protos = _mmd_fuse_protos(protos_g, protos_per)

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

    def _apply_train_mode(self, mode: str):
        for p in self.model.head.parameters():
            p.requires_grad = True
        for p in self.model_per.head.parameters():
            p.requires_grad = True

        if mode == "full":
            for p in self.model.base.parameters():
                p.requires_grad = True
            for p in self.model_per.base.parameters():
                p.requires_grad = True
        elif mode == "freeze_base":
            # Split-personalization: keep personalized feature extractor trainable.
            for p in self.model.base.parameters():
                p.requires_grad = False
            for p in self.model_per.base.parameters():
                p.requires_grad = True
        elif mode == "head_only":
            for p in self.model.base.parameters():
                p.requires_grad = False
            for p in self.model_per.base.parameters():
                p.requires_grad = False
        else:
            for p in self.model.base.parameters():
                p.requires_grad = True
            for p in self.model_per.base.parameters():
                p.requires_grad = True

    def _sync_personalized_from_global(self):
        a = float(max(0.0, min(1.0, self.personalized_sync_alpha)))
        for p_per, p_g in zip(self.model_per.base.parameters(), self.model.base.parameters()):
            p_per.data = a * p_per.data + (1.0 - a) * p_g.data

    def _normalize_entropy(self, entropy_value: float) -> float:
        denom = max(float(np.log(max(self.num_classes, 2))), 1e-8)
        return float(np.clip(entropy_value / denom, 0.0, 1.5))

    def _decide_training_mode(self) -> str:
        if self.server_contract_mode == "full":
            return "full"

        # Warmup: keep backbone learning active.
        if int(self.round_idx) < self.full_train_rounds:
            return "full"

        price = float(self.server_price_compute)
        ent = self._normalize_entropy(float(self.pred_entropy))
        drift = float(np.clip(self.proto_drift / max(2.0 * self.drift_keep_full_th, 1e-8), 0.0, 1.5))
        stale = float(np.clip(float(self.server_staleness) / max(self.stale_force_full_th, 1), 0.0, 1.5))
        price_norm = float(np.clip((price - 1.0) / max(self.head_only_price - 1.0, 1e-8), 0.0, 1.5))

        # Entropy/fuzzy-inspired scheduling: novelty keeps full training alive.
        full_score = 0.55 * ent + 0.35 * drift + 0.25 * stale - 0.45 * price_norm
        freeze_score = 0.45 * price_norm + 0.25 * ent + 0.25 * drift + 0.10 * stale
        head_score = 0.80 * price_norm + 0.20 * (1.0 - min(ent, 1.0)) + 0.15 * (1.0 - min(drift, 1.0)) - 0.20 * stale

        if self.force_affinity_update:
            full_score += 0.20

        if (self.server_staleness >= self.stale_force_full_th) or (ent >= self.entropy_keep_full_th) or (self.proto_drift >= self.drift_keep_full_th):
            if price >= self.head_only_price:
                return "freeze_base"
            return "full"

        if price >= self.head_only_price and head_score >= max(full_score, freeze_score):
            return "head_only"
        if price >= self.freeze_base_price and freeze_score >= full_score:
            return "freeze_base"
        return "full"

    def _maybe_update_affinity(self) -> bool:
        if self.force_affinity_update:
            return True

        drift = float(self.proto_drift)
        entropy_norm = self._normalize_entropy(float(self.pred_entropy))
        comm = float(self.server_price_comm)

        drift_score = (drift - self.affinity_drift_th) / max(self.affinity_drift_th, 1e-8)
        score = 2.0 * drift_score + 1.2 * (entropy_norm - 0.5) - 0.7 * (comm - 1.0)
        prob = 1.0 / (1.0 + np.exp(-score))
        prob = float(max(self.affinity_min_prob, min(1.0, prob)))

        return (np.random.rand() < prob) or (drift >= 2.0 * self.affinity_drift_th)

    def _build_delta_and_bid(self, prev_loss, base_before, val_loader, cls_cnt, round_train_cost):
        present = cls_cnt > 0
        if torch.any(present):
            self.tail_score = float(torch.mean(1.0 / torch.sqrt(cls_cnt[present] + 1.0)).item())
        else:
            self.tail_score = 0.0

        raw_delta = []
        for p_new, p_old in zip(self.model.base.parameters(), base_before):
            d = (p_new.detach() - p_old).clone()
            raw_delta.append(d)

        delta_list, sparse_bytes = self._compress_base_delta(raw_delta)
        sq = torch.tensor(0.0, device=self.device)
        non_zero = 0
        total_numel = 0
        for d in delta_list:
            sq += torch.sum(d * d)
            non_zero += int(torch.count_nonzero(d).item())
            total_numel += int(d.numel())
        self.delta_base = delta_list
        self.delta_norm = float(torch.sqrt(sq).item())
        dense_bytes = int(sum(d.numel() * d.element_size() for d in self.delta_base))
        self.delta_size = int(min(dense_bytes, sparse_bytes))
        self.delta_sparsity = 1.0 - float(non_zero / max(total_numel, 1))

        new_val_loss = float(self._recalculate_loss(self.model, val_loader, max_batches=self.bid_eval_batches)) if val_loader is not None else prev_loss
        loss_drop = float(max(prev_loss - new_val_loss, 0.0))
        self.last_val_loss = new_val_loss

        self.bid_loss_drop = float(loss_drop)
        self.bid_proto_drift = float(self.proto_drift)
        self.bid_staleness_term = float(self.server_staleness)

        hist_avg = round_train_cost
        if self.train_time_cost["num_rounds"] > 0:
            hist_avg = max(self.train_time_cost["total_cost"] / self.train_time_cost["num_rounds"], 1e-6)
        self.train_cost = float(round_train_cost)
        self.train_cost_ratio = float(round_train_cost / max(hist_avg, 1e-6))

        comm_cost = float(self.delta_size) / 1e6
        eff_price_compute = float(self.server_price_compute) * (1.0 - float(self.server_contract_discount_compute))
        eff_price_comm = float(self.server_price_comm) * (1.0 - float(self.server_contract_discount_comm))
        mf_penalty = abs(float(self.proto_drift) - float(self.server_mean_field_drift))
        entropy_norm = self._normalize_entropy(float(self.pred_entropy))
        stale_norm = float(np.clip(float(self.server_staleness) / max(self.stale_force_full_th, 1), 0.0, 1.0))
        drift_norm = float(np.clip(float(self.proto_drift) / max(2.0 * self.drift_keep_full_th, 1e-8), 0.0, 1.0))
        delta_strength = float(np.tanh(self.delta_norm))

        quality = 0.55 * self.bid_loss_drop + 0.20 * self.tail_score + 0.15 * delta_strength + 0.10 * self.val_confidence
        risk = 0.45 * entropy_norm + 0.35 * drift_norm + 0.20 * stale_norm
        credibility = (quality + 1e-6) * (1.0 + 0.25 * float(self.server_reputation)) / (1.0 + risk)
        novelty_bonus = self.uncertainty_bonus_weight * (0.60 * entropy_norm + 0.40 * drift_norm)
        time_penalty = self.compute_cost_weight * eff_price_compute * max(self.train_cost_ratio - 1.0, 0.0)
        comm_penalty = self.comm_cost_weight * eff_price_comm * comm_cost
        mode_penalty = 0.0 if self.last_train_mode == "full" else (0.15 if self.last_train_mode == "freeze_base" else 0.30)

        utility = (
            credibility
            + novelty_bonus
            + float(self.server_contract_bonus)
            + 0.10 * float(self.server_reputation)
            + 0.03 * float(self.server_staleness)
            - time_penalty
            - comm_penalty
            - 0.15 * mf_penalty
            - mode_penalty
        )

        self.quality_score = float(quality)
        self.risk_score = float(risk)
        self.credibility_score = float(credibility)
        self.delta_quality = float(self.bid_loss_drop / max(comm_cost, 1e-6)) if comm_cost > 0 else float(self.bid_loss_drop)
        self.bid_value = float(max(utility, 0.0))

    def _effective_delta_topk(self) -> float:
        if not self.enable_delta_compression:
            return 1.0

        if self.server_contract_mode == "full" and self.contract_force_dense_delta:
            return 1.0

        topk = float(np.clip(self.delta_topk, 1e-6, 1.0))
        if int(self.round_idx) < self.delta_topk_warmup_rounds:
            warmup_topk = float(np.clip(self.delta_topk_warmup_value, 1e-6, 1.0))
            topk = max(topk, warmup_topk)

        tail_factor = float(np.clip(self.tail_score, 0.0, 1.0))
        cred_factor = float(np.clip(self.credibility_score, 0.0, 2.0) / 2.0)
        topk = topk + self.delta_tail_boost * tail_factor + self.delta_cred_boost * cred_factor

        adaptive_cap = float(np.clip(self.delta_adaptive_topk_max, 1e-6, 1.0))
        adaptive_cap = max(adaptive_cap, float(np.clip(self.delta_topk, 1e-6, 1.0)))
        topk = min(topk, adaptive_cap)
        return float(np.clip(topk, 1e-6, 1.0))

    def _compress_base_delta(self, raw_delta):
        dense_bytes = int(sum(d.numel() * d.element_size() for d in raw_delta))
        topk = self._effective_delta_topk()
        if topk >= 1.0:
            self._delta_residual = [torch.zeros_like(d) for d in raw_delta]
            return [d.detach().clone() for d in raw_delta], dense_bytes

        if (self._delta_residual is None) or (len(self._delta_residual) != len(raw_delta)):
            self._delta_residual = [torch.zeros_like(d) for d in raw_delta]

        compressed = []
        sparse_bytes = 0
        for idx, delta in enumerate(raw_delta):
            corrected = delta
            if self.delta_error_feedback:
                corrected = corrected + self._delta_residual[idx]

            flat = corrected.reshape(-1)
            k = min(max(int(np.ceil(topk * flat.numel())), 1), flat.numel())
            if k >= flat.numel():
                sparse = corrected.clone()
                self._delta_residual[idx] = torch.zeros_like(delta)
                sparse_bytes += int(delta.numel() * delta.element_size())
            else:
                _, top_idx = torch.topk(flat.abs(), k, largest=True, sorted=False)
                sparse_flat = torch.zeros_like(flat)
                sparse_flat[top_idx] = flat[top_idx]
                sparse = sparse_flat.view_as(corrected)
                if self.delta_error_feedback:
                    self._delta_residual[idx] = (corrected - sparse).detach().clone()
                else:
                    self._delta_residual[idx] = torch.zeros_like(delta)
                sparse_bytes += int(k * (delta.element_size() + 4))
            compressed.append(sparse.detach().clone())

        return compressed, sparse_bytes

    def _build_candidate(self, recv_model):
        cand = copy.deepcopy(self.model)
        for p_c, p_r in zip(cand.base.parameters(), recv_model.base.parameters()):
            p_c.data.copy_(p_r.data)
        for p_c, p_h in zip(cand.head.parameters(), self.model.head.parameters()):
            p_c.data.copy_(p_h.data)
        return cand

    def _recalculate_loss_with_peer_base(self, peer_model, val_loader, max_batches=None):
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
                out = self.model.head(rep)
                loss = self.loss(out, y)
                total += float(loss.item())
                cnt += 1
                if max_batches is not None and (batch_idx + 1) >= max_batches:
                    break
        peer_model.base.train()
        self.model.head.train()
        return total / max(cnt, 1)

    def weight_cal(self, val_loader):
        if len(self.received_models) == 0:
            self.weight_vector = torch.zeros(self.num_clients, device=self.device)
            self._cached_baseline_loss = self._recalculate_loss(self.model, val_loader, max_batches=self.affinity_eval_batches)
            return torch.tensor([], device=self.device)

        weight_list = []
        baseline = self._recalculate_loss(self.model, val_loader, max_batches=self.affinity_eval_batches)
        self._cached_baseline_loss = baseline
        for received_model in self.received_models:
            gain = baseline - self._recalculate_loss_with_peer_base(
                received_model, val_loader, max_batches=self.affinity_eval_batches
            )
            params_dif = []
            for param_n, param_i in zip(received_model.base.parameters(), self.model.base.parameters()):
                params_dif.append((param_n - param_i).view(-1))
            params_dif = torch.cat(params_dif)
            weight_list.append(gain / (torch.norm(params_dif) + 1e-5))

        self._weight_vector_update(weight_list)
        if len(weight_list) == 0:
            return torch.tensor([], device=self.device)
        return torch.stack(weight_list)

    def _weight_vector_update(self, weight_list):
        vec = np.zeros(self.num_clients, dtype=np.float32)
        for w, cid in zip(weight_list, self.received_ids):
            vec[cid] += float(w.item())
        self.weight_vector = torch.tensor(vec, device=self.device)

    def weight_scale(self, weights: torch.Tensor):
        if weights.numel() == 0:
            return torch.tensor([], device=self.device)
        weights = torch.clamp_min(weights, 0.0)
        w_sum = torch.sum(weights)
        if float(w_sum.item()) > 0:
            return weights / w_sum
        return torch.tensor([], device=self.device)

    def add_parameters(self, w, received_model):
        for param, received_param in zip(self.model.base.parameters(), received_model.base.parameters()):
            param.data += received_param.data.clone() * float(w)

    def aggregate_parameters(self, val_loader):
        weights = self.weight_scale(self.weight_cal(val_loader))
        if len(weights) == 0:
            return

        mix = float(max(0.0, min(1.0, self.peer_mixing)))
        old_base = [p.data.clone() for p in self.model.base.parameters()]
        peer_base = [p.clone() for p in old_base]
        for w, received_model in zip(weights, self.received_models):
            for pb, ob, rp in zip(peer_base, old_base, received_model.base.parameters()):
                pb.add_((rp.data - ob) * float(w))
        for p, ob, pb in zip(self.model.base.parameters(), old_base, peer_base):
            p.data.copy_((1.0 - mix) * ob + mix * pb)

    def _recalculate_loss(self, new_model, val_loader, max_batches: Optional[int] = None):
        if val_loader is None:
            return 0.0
        new_model.eval()
        total = 0.0
        cnt = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x, y = self._to_device(x, y)
                out = new_model(x)
                loss = self.loss(out, y)
                total += float(loss.item())
                cnt += 1
                if max_batches is not None and (batch_idx + 1) >= max_batches:
                    break
        new_model.train()
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
                probs = F.softmax(eval_model(x), dim=1)
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

    def _to_device(self, x, y):
        if isinstance(x, list):
            x[0] = x[0].to(self.device)
            x = x[0]
        else:
            x = x.to(self.device)
        y = y.to(self.device)
        return x, y

    def train_metrics_personalized(self):
        trainloader, _ = self.load_train_data(for_eval=True)
        self.model_per.eval()
        train_num = 0
        losses = 0.0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(trainloader):
                x, y = self._to_device(x, y)
                output = self.model_per(x)
                loss = self.loss(output, y)
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
                output = self.model_per(x)
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
                output = self.model_per(x)
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
                output = self.model_per(x)
                y_true.append(y.detach().cpu().numpy())
                y_prob.append(F.softmax(output, dim=1).detach().cpu().numpy())
                if self.eval_calibration_max_batches > 0 and (batch_idx + 1) >= self.eval_calibration_max_batches:
                    break
        if len(y_true) == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0, self.num_classes), dtype=np.float32)
        y_true = np.concatenate(y_true, axis=0)
        y_prob = np.concatenate(y_prob, axis=0)
        return y_true, y_prob


def _proto_mean(protos: Dict[int, List[torch.Tensor]]):
    for label, proto_list in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for p in proto_list:
                proto += p.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]
    return protos


def _proto_mix_weight(cur: torch.Tensor, old: torch.Tensor, tau: float = 8.0, w_min: float = 0.15, w_max: float = 0.85):
    cur = F.normalize(cur, dim=0)
    old = F.normalize(old, dim=0)
    dist = torch.mean((cur - old) ** 2)
    return torch.exp(-tau * dist).clamp(w_min, w_max)


def _mmd_fuse_protos(protos_g: Dict[int, torch.Tensor], protos_per: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
    fused = {}
    labels = set(list(protos_g.keys()) + list(protos_per.keys()))
    for label in labels:
        if label in protos_g and label in protos_per:
            pg = protos_g[label]
            pp = protos_per[label]
            alpha = _proto_mix_weight(pg, pp)
            fused[label] = alpha * pg + (1.0 - alpha) * pp
        elif label in protos_g:
            fused[label] = protos_g[label]
        else:
            fused[label] = protos_per[label]
    return fused
