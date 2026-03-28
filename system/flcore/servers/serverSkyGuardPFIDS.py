import copy
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from flcore.clients.clientSkyGuardPFIDS import clientSkyGuardPFIDS
from flcore.servers.serverSGEFIDSUS import SGEFIDSUS
from utils.data_utils import load_dataset_config, read_public_data


class SkyGuardPFIDS(SGEFIDSUS):
    """Game-theoretic personalized FL IDS for low-altitude UAV swarms."""

    def __init__(self, args, times):
        super().__init__(args, times)

        self.clients = []
        self.set_clients(clientSkyGuardPFIDS)
        self.enable_contracts = False
        self.enable_sticky_sampling = False
        self.use_failover = False

        self.num_clusters = int(max(1, min(getattr(args, "skyguard_num_clusters", 3), self.num_clients)))
        self.adapter_momentum = float(getattr(args, "skyguard_adapter_momentum", 0.70))
        self.profile_momentum = float(getattr(args, "skyguard_profile_momentum", 0.80))
        self.replicator_lr = float(getattr(args, "skyguard_replicator_lr", 0.10))
        self.uncertainty_bonus = float(getattr(args, "skyguard_uncertainty_bonus", 0.05))
        self.prob_floor = float(getattr(args, "skyguard_prob_floor", 0.10))
        self.prob_ceiling = float(getattr(args, "skyguard_prob_ceiling", 0.90))
        self.online_floor = float(getattr(args, "skyguard_online_floor", 0.20))
        self.entropy_drift_th = float(getattr(args, "skyguard_entropy_drift_th", 0.70))
        self.proto_drift_th = float(getattr(args, "skyguard_proto_drift_th", 0.08))
        self.scheduler_prior_mix = float(getattr(args, "skyguard_scheduler_prior_mix", 0.10))
        self.sybil_similarity_th = float(getattr(args, "skyguard_sybil_similarity_th", 0.85))
        self.robust_min_factor = float(getattr(args, "skyguard_robust_min_factor", 0.10))
        self.dropout_ratio = float(getattr(args, "dropout_ratio", 0.0))
        self.comm_budget_mb = float(getattr(args, "comm_budget_mb", 0.0))
        if self.comm_budget_mb <= 0.0 and self.bandwidth_budget > 0:
            self.comm_budget_mb = float(self.bandwidth_budget) / (1024.0 * 1024.0)
        self.energy_budget = float(getattr(args, "energy_budget", 0.0))
        self.challenge_batches = int(max(1, getattr(args, "challenge_batches", getattr(args, "public_val_batches", 2))))
        self.server_verify_topk = int(max(0, getattr(args, "server_verify_topk", 0)))
        self.lambda_risk = float(getattr(args, "skyguard_lambda_risk", 0.18))
        self.cluster_fairness_bonus = float(getattr(args, "skyguard_cluster_fairness_bonus", 0.10))
        self.rare_label_bonus = float(getattr(args, "skyguard_rare_label_bonus", 0.10))
        self.require_challenge_data = bool(getattr(args, "skyguard_require_public_val", True))
        self.require_real_profiles = bool(getattr(args, "skyguard_require_real_profiles", True))

        self.participation_probs = torch.full((self.num_clients,), float(self.join_ratio), device=self.device)
        self.utility_mean = torch.zeros(self.num_clients, device=self.device)
        self.utility_var = torch.zeros(self.num_clients, device=self.device)
        self.global_delta_reference = None
        self._last_cluster_adapter_drift = 0.0
        self._global_drift_alert = False
        self._global_drift_pressure = 0.0
        self._stale_force_threshold = int(max(self.stale_priority_th, getattr(args, "stale_force_full_th", 12)))
        self._last_robust_weight_mean = 1.0
        self._last_robust_suspect_ratio = 0.0
        self._current_active_delta_stats = {}
        self._challenge_loss_baseline_cache = None
        self._last_cluster_coverage_ratio = 0.0
        self._last_rare_label_coverage_ratio = 0.0
        self._last_comm_used = 0.0
        self._last_energy_used = 0.0

        self.client_hist_matrix, self.client_clusters, self.client_profiles = self._load_or_build_uav_context()
        self.challenge_loader = self._load_public_challenge_loader()
        self.cluster_adapter_states = self._init_cluster_adapter_states()
        self._refresh_global_adapter_from_clusters()

        self.rs_participation_mean = []
        self.rs_participation_std = []
        self.rs_participation_entropy = []
        self.rs_online_ratio = []
        self.rs_uav_utility_mean = []
        self.rs_server_verified_utility_mean = []
        self.rs_server_challenge_gain_mean = []
        self.rs_local_gain_proxy_mean = []
        self.rs_grad_similarity_mean = []
        self.rs_data_quality_mean = []
        self.rs_resource_score_mean = []
        self.rs_attack_score_mean = []
        self.rs_cluster_adapter_drift = []
        self.rs_cluster_count = []
        self.rs_drift_alert_ratio = []
        self.rs_global_drift_pressure = []
        self.rs_robust_weight_mean = []
        self.rs_robust_suspect_ratio = []
        self.rs_comm_budget_util = []
        self.rs_energy_budget_util = []
        self.rs_cluster_coverage_ratio = []
        self.rs_rare_label_coverage_ratio = []

    def _dataset_config_path(self):
        return os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset", self.dataset, "config.json")

    def _load_or_build_uav_context(self):
        hist = np.zeros((self.num_clients, self.num_classes), dtype=np.float32)
        client_clusters = [0 for _ in range(self.num_clients)]
        client_profiles = {}
        cfg = load_dataset_config(self.dataset)

        stats = cfg.get("Size of samples for labels in clients", [])
        for cid in range(min(len(stats), self.num_clients)):
            for label, count in stats[cid]:
                label_int = int(label)
                if 0 <= label_int < self.num_classes:
                    hist[cid, label_int] = float(count)

        cfg_profiles = cfg.get("client_profiles", [])
        cfg_clusters = cfg.get("client_clusters", [])
        if not (isinstance(cfg_profiles, list) and len(cfg_profiles) >= self.num_clients):
            raise RuntimeError(
                f"SkyGuardPFIDS requires data-driven client_profiles in {self._dataset_config_path()}. "
                f"Please run the dataset preprocessor for '{self.dataset}' first."
            )

        if isinstance(cfg_clusters, list) and len(cfg_clusters) >= self.num_clients:
            client_clusters = [int(cfg_clusters[cid]) for cid in range(self.num_clients)]
        else:
            derived_clusters = []
            valid_cluster_field = True
            for cid in range(self.num_clients):
                profile = cfg_profiles[cid] if cid < len(cfg_profiles) else {}
                if not isinstance(profile, dict) or "cluster_id" not in profile:
                    valid_cluster_field = False
                    break
                derived_clusters.append(int(profile["cluster_id"]))
            if valid_cluster_field and len(derived_clusters) >= self.num_clients:
                client_clusters = derived_clusters[: self.num_clients]
            else:
                raise RuntimeError(
                    f"SkyGuardPFIDS requires client_clusters or profile.cluster_id entries in {self._dataset_config_path()}."
                )

        self.client_hist_matrix = hist
        for cid in range(self.num_clients):
            if not (isinstance(cfg_profiles[cid], dict)):
                raise RuntimeError(
                    f"SkyGuardPFIDS requires a real UAV profile for client {cid} in {self._dataset_config_path()}."
                )
            profile = dict(cfg_profiles[cid])
            profile["cluster_id"] = int(client_clusters[cid])
            profile.setdefault("availability", float(profile.get("expected_uptime", 0.8)))
            profile.setdefault("base_compute_capacity", float(profile.get("compute_capacity", 0.5)))
            profile.setdefault("base_link_quality", float(profile.get("link_quality", 0.5)))
            profile.setdefault("base_energy_level", float(profile.get("energy_level", 0.5)))
            client_profiles[cid] = profile

        return hist, client_clusters, client_profiles

    def _assign_clusters(self, hist: np.ndarray):
        if hist.shape[0] <= 0:
            return [0 for _ in range(self.num_clients)]
        totals = hist.sum(axis=1, keepdims=True)
        norm = hist / np.clip(totals, 1.0, None)
        ent = -np.sum(norm * np.log(np.clip(norm, 1e-12, 1.0)), axis=1, keepdims=True)
        ent = ent / max(np.log(max(self.num_classes, 2)), 1e-8)
        feats = np.concatenate([norm, ent], axis=1)
        cluster_count = int(max(1, min(self.num_clusters, feats.shape[0])))
        if cluster_count <= 1:
            return [0 for _ in range(self.num_clients)]
        try:
            return KMeans(
                n_clusters=cluster_count,
                random_state=int(getattr(self.args, "seed", 0)),
                n_init=10,
            ).fit_predict(feats).tolist()
        except Exception:
            return [int(i % cluster_count) for i in range(self.num_clients)]

    def _synthesize_profile(self, cid: int, hist_row: np.ndarray, cluster_id: int):
        raise RuntimeError(
            f"SkyGuardPFIDS is configured to use data-driven UAV context only. "
            f"Missing profile for client {cid} (cluster {cluster_id}) in dataset '{self.dataset}'."
        )

    def _init_cluster_adapter_states(self):
        if not hasattr(self.global_model, "adapter"):
            return {}
        base_state = {k: v.detach().cpu().clone() for k, v in self.global_model.adapter.state_dict().items()}
        return {int(cluster_id): copy.deepcopy(base_state) for cluster_id in sorted(set(self.client_clusters))}

    def _refresh_global_adapter_from_clusters(self):
        if not hasattr(self.global_model, "adapter") or len(self.cluster_adapter_states) == 0:
            return
        clusters = list(self.cluster_adapter_states.values())
        merged = {}
        for key in clusters[0].keys():
            stack = torch.stack([state[key].float() for state in clusters], dim=0)
            merged[key] = torch.mean(stack, dim=0)
        self.global_model.adapter.load_state_dict(merged, strict=True)

    def _assign_stackelberg_contracts(self):
        self.contract_clients = set()

    def _load_public_challenge_loader(self):
        public_data = read_public_data(self.dataset, split="public_val")
        if len(public_data) == 0:
            if self.require_challenge_data:
                raise RuntimeError(
                    f"SkyGuardPFIDS requires dataset/{self.dataset}/public_val.npz for server-side challenge validation. "
                    f"Please run the dataset preprocessor first."
                )
            return None
        return DataLoader(public_data, batch_size=self.batch_size, shuffle=False, drop_last=False)

    def _evaluate_challenge_loss(self, model, max_batches: int = None):
        if self.challenge_loader is None:
            return 0.0

        model = model.to(self.device)
        model.eval()
        total_loss = 0.0
        batch_count = 0
        max_batches = self.challenge_batches if max_batches is None else int(max_batches)
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.challenge_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                logits = model(x)
                total_loss += float(
                    F.cross_entropy(
                        logits,
                        y,
                        reduction="mean",
                        label_smoothing=float(max(self.args.label_smoothing, 0.0)),
                    ).item()
                )
                batch_count += 1
                if max_batches > 0 and (batch_idx + 1) >= max_batches:
                    break
        return total_loss / max(batch_count, 1)

    def _challenge_loss_baseline(self):
        if self._challenge_loss_baseline_cache is None:
            self._challenge_loss_baseline_cache = float(
                self._evaluate_challenge_loss(copy.deepcopy(self.global_model), max_batches=self.challenge_batches)
            )
        return float(self._challenge_loss_baseline_cache)

    def _client_payment_cost(self, client):
        delta_mb = max(float(getattr(client, "delta_size", 0)) / (1024.0 * 1024.0), 1e-6)
        energy_cost = float(getattr(client, "mode_compute_cost", getattr(client, "train_cost_ratio", 1.0)))
        return float(self.price_comm * delta_mb + self.price_compute * energy_cost)

    def _estimate_energy_cost(self, client):
        return float(getattr(client, "mode_compute_cost", getattr(client, "train_cost_ratio", 1.0)))

    def _compute_active_delta_stats(self, active_clients: List[object]):
        norms = []
        for client in active_clients:
            if getattr(client, "delta_base", None) is None:
                continue
            norms.append(float(getattr(client, "delta_norm", 0.0)))

        if len(norms) == 0:
            return {"median_norm": 0.0, "mad_norm": 1.0}

        median_norm = float(np.median(norms))
        mad_norm = float(np.median(np.abs(np.asarray(norms) - median_norm)))
        return {
            "median_norm": median_norm,
            "mad_norm": max(mad_norm, 1e-6),
        }

    def _rare_labels(self):
        if self.client_hist_matrix.size == 0:
            return []
        support = np.asarray(self.client_hist_matrix.sum(axis=0), dtype=np.float64)
        positive = support[support > 0]
        if positive.size == 0:
            return []
        threshold = float(np.quantile(positive, 0.35))
        return [int(label) for label, count in enumerate(support.tolist()) if count > 0 and count <= threshold]

    def _rare_label_score(self, cid: int):
        rare_labels = self._rare_labels()
        if len(rare_labels) == 0:
            return 0.0
        hist_row = self.client_hist_matrix[cid]
        covered = sum(1 for label in rare_labels if label < len(hist_row) and hist_row[label] > 0)
        return float(covered / max(len(rare_labels), 1))

    def _normalize_entropy_value(self, entropy_value: float) -> float:
        denom = max(float(np.log(max(self.num_classes, 2))), 1e-8)
        return float(np.clip(entropy_value / denom, 0.0, 1.5))

    def _update_global_drift_state(self):
        mean_entropy = float(self.rs_mean_pred_entropy[-1]) if len(self.rs_mean_pred_entropy) > 0 else 0.0
        mean_proto_drift = float(self.rs_proto_drift_mean[-1]) if len(self.rs_proto_drift_mean) > 0 else 0.0
        entropy_norm = self._normalize_entropy_value(mean_entropy)
        drift_norm = float(np.clip(mean_proto_drift / max(self.proto_drift_th, 1e-8), 0.0, 1.5))
        pressure = float(np.clip(0.60 * entropy_norm + 0.40 * drift_norm, 0.0, 1.5))

        if pressure >= 1.00:
            self._global_drift_alert = True
        elif pressure <= 0.80:
            self._global_drift_alert = False
        self._global_drift_pressure = pressure
        return pressure, self._global_drift_alert

    def _weighted_sample(self, candidates: List[int], weights: np.ndarray, k: int):
        if k <= 0 or len(candidates) == 0:
            return []
        k = min(k, len(candidates))
        weights = np.asarray(weights, dtype=np.float64)
        weights = np.clip(weights, 0.0, None)
        if float(weights.sum()) <= 0.0:
            return random.sample(candidates, k)
        probs = weights / weights.sum()
        chosen = np.random.choice(np.asarray(candidates), size=k, replace=False, p=probs)
        return [int(x) for x in chosen.tolist()]

    def _evolve_uav_profiles(self, selected_ids: List[int]):
        selected_set = set(int(cid) for cid in selected_ids)
        momentum = float(np.clip(self.profile_momentum, 0.0, 0.99))
        for cid in range(self.num_clients):
            profile = self.client_profiles[cid]
            selected = cid in selected_set
            base_compute = float(profile.get("base_compute_capacity", profile.get("compute_capacity", 0.5)))
            base_link = float(profile.get("base_link_quality", profile.get("link_quality", 0.5)))
            base_energy = float(profile.get("base_energy_level", profile.get("energy_level", 0.5)))
            mobility_score = float(np.clip(profile.get("mobility_score", profile.get("support_ratio", 0.5)), 0.0, 1.0))
            burstiness = float(np.clip(profile.get("burstiness", 0.0), 0.0, 1.0))
            stability = float(np.clip(1.0 - burstiness, 0.0, 1.0))
            energy = float(profile.get("energy_level", base_energy))
            energy_drain = 0.05 if selected else 0.01
            energy_recover = 0.02 if (not selected) else 0.0
            energy = float(np.clip(energy - energy_drain + energy_recover, 0.05, 0.99))
            link = float(
                np.clip(
                    momentum * float(profile.get("link_quality", base_link))
                    + (1.0 - momentum) * (0.55 * base_link + 0.25 * mobility_score + 0.20 * stability),
                    0.05,
                    0.99,
                )
            )
            compute = float(np.clip(0.60 * base_compute + 0.25 * energy + 0.15 * stability, 0.08, 0.99))
            expected_uptime = float(profile.get("expected_uptime", profile.get("availability", 0.7)))
            availability = float(
                np.clip(
                    0.25 * expected_uptime + 0.25 * energy + 0.25 * link + 0.25 * compute
                    - self.dropout_ratio * float(profile.get("dropout_risk", 0.0)),
                    self.online_floor,
                    0.99,
                )
            )

            profile["energy_level"] = energy
            profile["link_quality"] = link
            profile["compute_capacity"] = compute
            profile["availability"] = availability
            profile["expected_uptime"] = availability
            profile["dropout_risk"] = float(np.clip(1.0 - availability, 0.01, 0.95))

    def _broadcast_round_context(self, round_idx: int):
        super()._broadcast_round_context(round_idx)
        drift_pressure, drift_alert = self._update_global_drift_state()
        delta_ref = None if self.global_delta_reference is None else [d.detach().cpu().clone() for d in self.global_delta_reference]
        for client in self.clients:
            cid = int(client.id)
            cluster_id = int(self.client_clusters[cid])
            if hasattr(client, "receive_cluster_adapter"):
                client.receive_cluster_adapter(self.cluster_adapter_states.get(cluster_id), cluster_id)
            if hasattr(client, "receive_uav_context"):
                client.receive_uav_context(
                    profile=self.client_profiles[cid],
                    cluster_id=cluster_id,
                    participation_prob=float(self.participation_probs[cid].item()),
                    global_delta_reference=delta_ref,
                    drift_alert=drift_alert,
                    drift_pressure=float(drift_pressure),
                )

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(
                range(self.num_join_clients, self.num_clients + 1), 1, replace=False
            )[0]
        else:
            self.current_num_join_clients = self.num_join_clients

        target = int(max(1, self.current_num_join_clients))
        probs = self.participation_probs.detach().cpu().numpy().astype(np.float64)
        probs = np.clip(probs, self.prob_floor, self.prob_ceiling)
        stale_values = self.staleness.detach().cpu().numpy().astype(np.float64)
        stale_boost = 1.0 + self.stale_score_alpha * np.clip(stale_values / max(self.stale_priority_th, 1), 0.0, 2.0)
        forced_mask = stale_values >= float(self._stale_force_threshold)
        probs = probs * stale_boost
        probs[forced_mask] *= 1.10

        chosen = self._weighted_sample(list(range(self.num_clients)), probs, target)

        if len(chosen) < target:
            fallback = [cid for cid in range(self.num_clients) if cid not in chosen]
            fallback_scores = np.asarray([probs[cid] for cid in fallback], dtype=np.float64)
            chosen += self._weighted_sample(fallback, fallback_scores, target - len(chosen))

        self._evolve_uav_profiles(chosen)
        return [self.clients[int(cid)] for cid in chosen]

    def _draw_active_clients(self, selected_clients: List[object]):
        active_clients = []
        p_online = []
        for client in selected_clients:
            cid = int(client.id)
            profile = self.client_profiles[cid]
            online_prob = float(
                np.clip(
                    profile.get("availability", profile.get("expected_uptime", 1.0))
                    * (0.75 + 0.25 * float(self.participation_probs[cid].item())),
                    self.online_floor,
                    0.995,
                )
            )
            p_online.append(online_prob)
            if np.random.rand() < online_prob:
                active_clients.append(client)

        max_active = max(1, int(round(self.client_activity_rate * max(len(selected_clients), 1))))
        if len(active_clients) == 0 and len(selected_clients) > 0:
            best_idx = int(np.argmax(np.asarray(p_online, dtype=np.float64)))
            active_clients = [selected_clients[best_idx]]
        elif len(active_clients) > max_active:
            active_ids = [client.id for client in active_clients]
            active_scores = np.asarray([self.client_profiles[int(cid)]["availability"] for cid in active_ids], dtype=np.float64)
            keep_ids = set(self._weighted_sample(active_ids, active_scores, max_active))
            active_clients = [client for client in active_clients if client.id in keep_ids]
        return active_clients

    def _trust_weight(self, client):
        cid = int(client.id)
        rep = float(self.reputation[cid].item())
        stale = float(self.staleness[cid].item())
        attack = float(getattr(client, "attack_score", getattr(client, "risk_score", 0.0)))
        quality_signal = float(
            np.clip(
                max(float(getattr(client, "server_verified_value", 0.0)), float(getattr(client, "quality_score", 0.0))),
                0.0,
                1.0,
            )
        )
        stale_decay = float(np.exp(-self.staleness_decay_gamma * max(stale, 0.0)))
        return rep * stale_decay * (0.5 + 0.5 * quality_signal) / (1.0 + max(attack, 0.0))

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

    def _reset_server_round_signals(self, clients: List[object]):
        for client in clients:
            client.server_verified_gain = 0.0
            client.server_verified_value = 0.0
            client.server_grad_similarity_score = 0.0
            client.server_challenge_loss = float("inf")
            client.server_comm_cost = 0.0
            client.server_energy_cost = 0.0
            client.server_risk_score = 0.0

    def _verification_candidates(self, active_clients: List[object]) -> List[object]:
        usable = [
            client
            for client in active_clients
            if getattr(client, "delta_base", None) is not None and float(getattr(client, "delta_norm", 0.0)) >= self.delta_min_norm
        ]
        if self.server_verify_topk <= 0 or len(usable) <= self.server_verify_topk:
            return usable
        usable.sort(
            key=lambda client: (
                float(getattr(client, "bid_value", 0.0)),
                float(getattr(client, "local_gain_proxy", 0.0)),
                float(getattr(client, "quality_score", 0.0)),
            ),
            reverse=True,
        )
        return usable[: self.server_verify_topk]

    def _server_verified_value(self, client):
        cid = int(client.id)
        if getattr(client, "delta_base", None) is None:
            client.server_verified_value = -1e9
            return float(client.server_verified_value)

        base_loss = self._challenge_loss_baseline()
        shadow_model = copy.deepcopy(self.global_model).to(self.device)
        for param, delta in zip(shadow_model.base.parameters(), client.delta_base):
            param.data.add_(delta.to(self.device), alpha=float(self.delta_server_lr))

        new_loss = float(self._evaluate_challenge_loss(shadow_model, max_batches=self.challenge_batches))
        gain = float(max(base_loss - new_loss, 0.0))
        delta_mb = max(float(getattr(client, "delta_size", 0)) / (1024.0 * 1024.0), 1e-6)
        energy = float(getattr(client, "mode_compute_cost", getattr(client, "train_cost_ratio", 1.0)))
        risk = float(getattr(client, "risk_score", getattr(client, "attack_score", 0.0)))
        grad_align = float(self._delta_similarity(getattr(client, "delta_base", None), self.global_delta_reference))
        verified_value = gain - self.price_comm * delta_mb - self.price_compute * energy - self.lambda_risk * risk

        client.server_grad_similarity_score = float(grad_align)
        client.server_verified_gain = float(gain)
        client.server_challenge_gain = float(gain)
        client.server_challenge_loss = float(new_loss)
        client.server_comm_cost = float(delta_mb)
        client.server_energy_cost = float(energy)
        client.server_risk_score = float(risk)
        client.server_verified_value = float(verified_value)
        return float(verified_value)

    def _participation_entropy(self):
        prob = self.participation_probs.detach().cpu().numpy().astype(np.float64)
        prob = np.clip(prob, self.prob_floor, self.prob_ceiling)
        prob = prob / max(float(np.sum(prob)), 1e-12)
        entropy = -np.sum(prob * np.log(np.clip(prob, 1e-12, 1.0)))
        return float(entropy / max(np.log(max(self.num_clients, 2)), 1e-8))

    def _robust_client_factors(self):
        n_up = len(self.uploaded_deltas)
        if n_up <= 1:
            self._last_robust_weight_mean = 1.0
            self._last_robust_suspect_ratio = 0.0
            return np.ones(n_up, dtype=np.float64)

        flat_updates = []
        for delta_list in self.uploaded_deltas:
            flat_updates.append(torch.cat([d.detach().float().view(-1).cpu() for d in delta_list], dim=0))
        stacked = torch.stack(flat_updates, dim=0)

        norms = torch.linalg.norm(stacked, dim=1).clamp_min(1e-12)
        unit = stacked / norms.unsqueeze(1)
        sim = torch.matmul(unit, unit.T)
        sim.fill_diagonal_(0.0)
        mean_sim = sim.sum(dim=1) / max(n_up - 1, 1)

        median_vec = torch.median(stacked, dim=0).values
        median_norm = torch.linalg.norm(median_vec).clamp_min(1e-12)
        deviation = torch.linalg.norm(stacked - median_vec.unsqueeze(0), dim=1) / median_norm
        norm_ratio = norms / torch.median(norms).clamp_min(1e-12)

        sybil_excess = torch.clamp(mean_sim - self.sybil_similarity_th, min=0.0)
        dev_excess = torch.clamp(deviation - torch.median(deviation), min=0.0)
        norm_excess = torch.clamp(norm_ratio - 2.5, min=0.0)
        penalty = 1.50 * sybil_excess + 0.75 * dev_excess + 0.75 * norm_excess
        factors = torch.exp(-penalty).clamp(min=self.robust_min_factor, max=1.0)

        self._last_robust_weight_mean = float(torch.mean(factors).item())
        self._last_robust_suspect_ratio = float(torch.mean((factors < 0.5).float()).item())
        return factors.detach().cpu().numpy().astype(np.float64)

    def _select_budgeted_winners(self, active_clients: List[object]) -> List[object]:
        if len(active_clients) == 0:
            return []

        self._reset_server_round_signals(active_clients)
        candidate_clients = self._verification_candidates(active_clients)
        records = []
        for client in candidate_clients:
            cid = int(client.id)
            value = self._server_verified_value(client)
            if value <= 0.0:
                continue
            comm_mb = max(float(getattr(client, "server_comm_cost", 0.0)), 1e-6)
            energy = max(float(getattr(client, "server_energy_cost", 0.0)), 1e-6)
            rare_score = self._rare_label_score(cid)
            ratio = value / max(comm_mb + energy, 1e-6)
            records.append(
                {
                    "client": client,
                    "cid": cid,
                    "cluster_id": int(self.client_clusters[cid]),
                    "value": float(value),
                    "ratio": float(ratio),
                    "comm_mb": float(comm_mb),
                    "energy": float(energy),
                    "rare_score": float(rare_score),
                }
            )

        if len(records) == 0:
            self._last_comm_used = 0.0
            self._last_energy_used = 0.0
            self._last_cluster_coverage_ratio = 0.0
            self._last_rare_label_coverage_ratio = 0.0
            return []

        records.sort(key=lambda item: (item["ratio"], item["value"]), reverse=True)
        winners = []
        used_comm = 0.0
        used_energy = 0.0
        covered_clusters = set()
        for record in records:
            next_comm = used_comm + float(record["comm_mb"])
            next_energy = used_energy + float(record["energy"])
            if self.comm_budget_mb > 0 and next_comm > float(self.comm_budget_mb):
                continue
            if self.energy_budget > 0 and next_energy > float(self.energy_budget):
                continue

            coverage_bonus = self.cluster_fairness_bonus if int(record["cluster_id"]) not in covered_clusters else 0.0
            if record["value"] + coverage_bonus + self.rare_label_bonus * record["rare_score"] <= 0.0:
                continue

            winners.append(record["client"])
            used_comm = next_comm
            used_energy = next_energy
            covered_clusters.add(int(record["cluster_id"]))

        if len(winners) == 0 and len(records) > 0:
            winners = [records[0]["client"]]
            used_comm = float(records[0]["comm_mb"])
            used_energy = float(records[0]["energy"])

        winner_ids = {int(client.id) for client in winners}
        active_clusters = {int(self.client_clusters[int(client.id)]) for client in active_clients}
        winner_clusters = {int(self.client_clusters[int(client.id)]) for client in winners}
        rare_labels = self._rare_labels()
        rare_label_covered = set()
        for cid in winner_ids:
            row = self.client_hist_matrix[cid]
            for label in rare_labels:
                if label < len(row) and row[label] > 0:
                    rare_label_covered.add(int(label))

        self._last_comm_used = float(used_comm)
        self._last_energy_used = float(used_energy)
        self._last_cluster_coverage_ratio = float(len(winner_clusters) / max(len(active_clusters), 1))
        self._last_rare_label_coverage_ratio = (
            float(len(rare_label_covered) / max(len(rare_labels), 1)) if len(rare_labels) > 0 else 1.0
        )
        return winners

    def _select_auction_winners(self, active_clients: List[object]) -> List[object]:
        return self._select_budgeted_winners(active_clients)

    def _aggregate_cluster_adapters(self, clients: List[object]):
        if not hasattr(self.global_model, "adapter") or len(clients) == 0:
            self._last_cluster_adapter_drift = 0.0
            return

        grouped = defaultdict(list)
        for client in clients:
            cid = int(client.id)
            cluster_id = int(self.client_clusters[cid])
            adapter_state = getattr(client, "export_cluster_adapter", lambda: {})()
            if len(adapter_state) == 0:
                continue
            weight = float(client.train_samples) * float(self.participation_probs[cid].item()) * self._trust_weight(client)
            grouped[cluster_id].append((adapter_state, max(weight, 1e-6)))

        drift_vals = []
        momentum = float(np.clip(self.adapter_momentum, 0.0, 0.99))
        for cluster_id, items in grouped.items():
            total_weight = float(sum(weight for _, weight in items))
            if total_weight <= 0.0:
                continue

            old_state = self.cluster_adapter_states.get(cluster_id, None)
            new_state = {}
            for key in items[0][0].keys():
                agg = None
                for state, weight in items:
                    val = state[key].detach().cpu().float()
                    if agg is None:
                        agg = val * weight
                    else:
                        agg += val * weight
                agg = agg / total_weight
                if old_state is not None and key in old_state:
                    prev = old_state[key].detach().cpu().float()
                    drift_vals.append(float(torch.mean(torch.abs(agg - prev)).item()))
                    agg = momentum * prev + (1.0 - momentum) * agg
                new_state[key] = agg
            self.cluster_adapter_states[cluster_id] = new_state

        self._last_cluster_adapter_drift = float(np.mean(drift_vals)) if len(drift_vals) > 0 else 0.0
        self._refresh_global_adapter_from_clusters()

    def _update_participation_probabilities(self, effective_winners: List[object], selected_clients: List[object]):
        if self.num_clients <= 0:
            return

        base_prob = self.participation_probs.detach().cpu().numpy().astype(np.float64)
        base_prob = np.clip(base_prob, self.prob_floor, self.prob_ceiling)
        prior_prob = np.asarray(
            [
                float(np.clip(self.client_profiles[cid].get("availability", 0.5), self.prob_floor, self.prob_ceiling))
                for cid in range(self.num_clients)
            ],
            dtype=np.float64,
        )
        prior_mix = float(np.clip(self.scheduler_prior_mix, 0.0, 0.5))
        selected_ids = {int(client.id) for client in selected_clients}
        winner_ids = {int(client.id) for client in effective_winners}
        uploaded_ids = {int(cid) for cid in self.uploaded_ids}
        payoffs = np.zeros(self.num_clients, dtype=np.float64)

        for cid in range(self.num_clients):
            stale = float(self.staleness[cid].item())
            stale_bonus = 0.15 * min(stale / max(self._stale_force_threshold, 1), 1.5)
            if cid in winner_ids:
                client = self.clients[cid]
                realized = float(getattr(client, "server_verified_value", 0.0))
                uncertainty = float(getattr(client, "utility_uncertainty", 0.0))
                payoff = realized + self.uncertainty_bonus * uncertainty + stale_bonus
                self.utility_mean[cid] = 0.8 * self.utility_mean[cid] + 0.2 * realized
                dev = realized - float(self.utility_mean[cid].item())
                self.utility_var[cid] = 0.8 * self.utility_var[cid] + 0.2 * (dev * dev)
            elif cid in selected_ids:
                idle_penalty = -0.40 if cid not in uploaded_ids else -0.15
                payoff = idle_penalty + stale_bonus
            else:
                payoff = 0.05 * stale_bonus
            payoffs[cid] = float(np.clip(payoff, -4.0, 4.0))

        weights = base_prob * np.exp(self.replicator_lr * payoffs)
        weights = (1.0 - prior_mix) * weights + prior_mix * prior_prob
        weights = np.clip(weights, 1e-12, None)
        weights = weights / max(float(np.sum(weights)), 1e-12)
        scaled_prob = weights * float(self.num_clients) * float(self.join_ratio)
        scaled_prob = np.clip(scaled_prob, self.prob_floor, self.prob_ceiling)
        self.participation_probs = torch.as_tensor(
            scaled_prob,
            device=self.device,
            dtype=self.participation_probs.dtype,
        )

    def receive_models(self, round_idx: int):
        assert len(self.selected_clients) > 0

        self._challenge_loss_baseline_cache = None
        active_clients = self._draw_active_clients(self.selected_clients)
        self._current_active_delta_stats = self._compute_active_delta_stats(active_clients)
        winners = self._select_budgeted_winners(active_clients)
        winner_ids = {int(c.id) for c in winners}

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_protos = []
        self.uploaded_deltas = []
        self.uploaded_delta_sizes = []

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
        local_utility_values = []
        verified_value_values = []
        challenge_gain_values = []
        local_gain_values = []
        grad_values = []
        quality_values = []
        resource_values = []
        attack_values = []
        drift_alert_values = []
        mode_full = 0
        mode_freeze = 0
        mode_head = 0
        successful_winner_clients = []

        reject_time = 0
        reject_missing_delta = 0
        reject_below_norm = 0

        tot_samples = 0
        for client in active_clients:
            verified_value = float(getattr(client, "server_verified_value", 0.0))
            bid_values.append(float(getattr(client, "bid_value", 0.0)))
            proto_drift_values.append(float(getattr(client, "proto_drift", 0.0)))
            bid_loss_drop_values.append(float(getattr(client, "bid_loss_drop", 0.0)))
            bid_drift_values.append(float(getattr(client, "bid_proto_drift", getattr(client, "proto_drift", 0.0))))
            bid_staleness_values.append(float(getattr(client, "bid_staleness_term", 0.0)))
            pred_entropy_values.append(float(getattr(client, "pred_entropy", 0.0)))
            credibility_values.append(float(getattr(client, "credibility_score", 0.0)))
            delta_sparsity_values.append(float(getattr(client, "delta_sparsity", 0.0)))
            local_utility_values.append(float(getattr(client, "local_strategy_score", 0.0)))
            verified_value_values.append(float(verified_value))
            challenge_gain_values.append(float(getattr(client, "server_verified_gain", 0.0)))
            local_gain_values.append(float(getattr(client, "local_gain_proxy", 0.0)))
            grad_values.append(
                float(getattr(client, "server_grad_similarity_score", getattr(client, "grad_similarity_score", 0.0)))
            )
            quality_values.append(float(getattr(client, "data_quality_score", 0.0)))
            resource_values.append(float(getattr(client, "resource_score", 0.0)))
            attack_values.append(float(getattr(client, "attack_score", 0.0)))
            drift_alert_values.append(float(getattr(client, "drift_alert", 0.0)))

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

            if int(client.id) not in winner_ids:
                self.staleness[client.id] += 1
                drift = float(getattr(client, "proto_drift", 0.0))
                if drift > float(getattr(self.args, "affinity_drift_th", 0.0)):
                    self._reward_reputation(client.id, good=False, mild=True)
                continue

            try:
                client_time_cost = (
                    client.train_time_cost["total_cost"] / client.train_time_cost["num_rounds"]
                    + client.send_time_cost["total_cost"] / client.send_time_cost["num_rounds"]
                )
            except ZeroDivisionError:
                client_time_cost = 0.0

            if client_time_cost > self.time_threthold:
                reject_time += 1
                self.staleness[client.id] += 1
                self._reward_reputation(client.id, good=False)
                continue
            if getattr(client, "delta_base", None) is None:
                reject_missing_delta += 1
                self.staleness[client.id] += 1
                self._reward_reputation(client.id, good=False)
                continue

            delta_norm = float(getattr(client, "delta_norm", 0.0))
            if delta_norm < self.delta_min_norm:
                reject_below_norm += 1
                self.staleness[client.id] += 1
                self._reward_reputation(client.id, good=False)
                continue

            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_deltas.append(client.delta_base)
            self.uploaded_delta_sizes.append(int(getattr(client, "delta_size", 0)))
            delta_norm_values.append(delta_norm)
            successful_winner_clients.append(client)

            for cc in client.protos.keys():
                y = torch.tensor(cc, dtype=torch.int64)
                self.uploaded_protos.append((client.protos[cc], y))

            if getattr(client, "affinity_updated", True):
                affinity_update_count += 1
                scale = float(self.reputation[client.id].clamp(min=0.1, max=2.0).item())
                self.P[client.id] += client.weight_vector * scale

            self.client_models[client.id] = copy.deepcopy(client.model)
            self.staleness[client.id] = 0
            self._reward_reputation(client.id, good=True)

        if tot_samples > 0:
            for i, w in enumerate(self.uploaded_weights):
                self.uploaded_weights[i] = w / tot_samples
        elif len(self.uploaded_weights) > 0:
            norm = float(sum(self.uploaded_weights))
            for i, w in enumerate(self.uploaded_weights):
                self.uploaded_weights[i] = w / max(norm, 1e-12)

        effective_winner_ids = {int(cid) for cid in self.uploaded_ids}
        effective_winners = [client for client in successful_winner_clients if int(client.id) in effective_winner_ids]
        self._aggregate_global_prototypes(effective_winners)
        self._aggregate_cluster_adapters(effective_winners)
        self._update_participation_probabilities(effective_winners, self.selected_clients)

        proto_bytes = 0
        for p, _ in self.uploaded_protos:
            try:
                proto_bytes += int(p.numel() * p.element_size())
            except Exception:
                pass

        delta_bytes = int(sum(self.uploaded_delta_sizes))
        comm_budget_bytes = int(round(self.comm_budget_mb * 1024.0 * 1024.0)) if self.comm_budget_mb > 0 else 0

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
        self.rs_bandwidth_budget_bytes.append(float(comm_budget_bytes))
        self.rs_bandwidth_utilization.append(
            float(delta_bytes / comm_budget_bytes) if comm_budget_bytes > 0 else 0.0
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
        self.rs_failover_used_count.append(0.0)
        self.rs_failover_round_flag.append(0.0)
        self.rs_contract_clients.append(0.0)
        self.rs_contract_active_ratio.append(0.0)
        self.rs_contract_winner_ratio.append(0.0)
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

        self.rs_participation_mean.append(float(torch.mean(self.participation_probs).item()))
        self.rs_participation_std.append(float(torch.std(self.participation_probs, unbiased=False).item()))
        self.rs_participation_entropy.append(float(self._participation_entropy()))
        self.rs_online_ratio.append(float(active_cnt / max(selected_cnt, 1)))
        self.rs_uav_utility_mean.append(float(np.mean(local_utility_values)) if len(local_utility_values) > 0 else 0.0)
        self.rs_server_verified_utility_mean.append(
            float(np.mean(verified_value_values)) if len(verified_value_values) > 0 else 0.0
        )
        self.rs_server_challenge_gain_mean.append(
            float(np.mean(challenge_gain_values)) if len(challenge_gain_values) > 0 else 0.0
        )
        self.rs_local_gain_proxy_mean.append(float(np.mean(local_gain_values)) if len(local_gain_values) > 0 else 0.0)
        self.rs_grad_similarity_mean.append(float(np.mean(grad_values)) if len(grad_values) > 0 else 0.0)
        self.rs_data_quality_mean.append(float(np.mean(quality_values)) if len(quality_values) > 0 else 0.0)
        self.rs_resource_score_mean.append(float(np.mean(resource_values)) if len(resource_values) > 0 else 0.0)
        self.rs_attack_score_mean.append(float(np.mean(attack_values)) if len(attack_values) > 0 else 0.0)
        self.rs_cluster_adapter_drift.append(float(self._last_cluster_adapter_drift))
        self.rs_cluster_count.append(float(len(set(self.client_clusters))))
        self.rs_drift_alert_ratio.append(float(np.mean(drift_alert_values)) if len(drift_alert_values) > 0 else 0.0)
        self.rs_global_drift_pressure.append(float(self._global_drift_pressure))
        self.rs_robust_weight_mean.append(float(self._last_robust_weight_mean))
        self.rs_robust_suspect_ratio.append(float(self._last_robust_suspect_ratio))
        self.rs_comm_budget_util.append(float(self._last_comm_used / self.comm_budget_mb) if self.comm_budget_mb > 0 else 0.0)
        self.rs_energy_budget_util.append(float(self._last_energy_used / self.energy_budget) if self.energy_budget > 0 else 0.0)
        self.rs_cluster_coverage_ratio.append(float(self._last_cluster_coverage_ratio))
        self.rs_rare_label_coverage_ratio.append(float(self._last_rare_label_coverage_ratio))

    def _aggregate_base_deltas(self):
        if len(self.uploaded_deltas) == 0:
            if len(self.rs_robust_weight_mean) > 0:
                self.rs_robust_weight_mean[-1] = 1.0
                self.rs_robust_suspect_ratio[-1] = 0.0
            return

        base_params = list(self.global_model.base.parameters())
        agg = [torch.zeros_like(p.data) for p in base_params]
        robust_factor = self._robust_client_factors()

        rep_weight = []
        for i, cid in enumerate(self.uploaded_ids):
            cid_int = int(cid)
            client = self.clients[cid_int]
            participation = float(self.participation_probs[cid_int].item())
            weight_i = (
                float(self.uploaded_weights[i])
                * max(participation, self.prob_floor)
                * self._trust_weight(client)
                * float(robust_factor[i] if i < len(robust_factor) else 1.0)
            )
            rep_weight.append(weight_i)
        rep_weight = np.asarray(rep_weight, dtype=np.float64)
        rep_sum = float(np.sum(rep_weight))
        if rep_sum <= 0.0:
            rep_weight = np.asarray(self.uploaded_weights, dtype=np.float64)
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

        self.global_delta_reference = [d.detach().cpu().clone() for d in agg]
        if len(self.rs_robust_weight_mean) > 0:
            self.rs_robust_weight_mean[-1] = float(self._last_robust_weight_mean)
            self.rs_robust_suspect_ratio[-1] = float(self._last_robust_suspect_ratio)

        lr = float(max(0.0, self.delta_server_lr))
        for p, d in zip(base_params, agg):
            p.data += d * lr

    def _print_round_resource_metrics(self):
        super()._print_round_resource_metrics()
        if len(self.rs_uav_utility_mean) == 0:
            return
        print(
            "[SkyGuard Metrics] "
            f"participation_mean={self.rs_participation_mean[-1]:.4f}, "
            f"participation_std={self.rs_participation_std[-1]:.4f}, "
            f"participation_entropy={self.rs_participation_entropy[-1]:.4f}, "
            f"verified_utility={self.rs_server_verified_utility_mean[-1]:.4f}, "
            f"challenge_gain={self.rs_server_challenge_gain_mean[-1]:.4f}, "
            f"robust_weight={self.rs_robust_weight_mean[-1]:.4f}, "
            f"robust_suspect={self.rs_robust_suspect_ratio[-1]:.4f}, "
            f"local_gain={self.rs_local_gain_proxy_mean[-1]:.4f}, "
            f"grad_similarity={self.rs_grad_similarity_mean[-1]:.4f}, "
            f"data_quality={self.rs_data_quality_mean[-1]:.4f}, "
            f"resource_score={self.rs_resource_score_mean[-1]:.4f}, "
            f"attack_score={self.rs_attack_score_mean[-1]:.4f}, "
            f"cluster_coverage={self.rs_cluster_coverage_ratio[-1]:.4f}, "
            f"rare_label_coverage={self.rs_rare_label_coverage_ratio[-1]:.4f}, "
            f"comm_budget_util={self.rs_comm_budget_util[-1]:.4f}, "
            f"energy_budget_util={self.rs_energy_budget_util[-1]:.4f}, "
            f"adapter_drift={self.rs_cluster_adapter_drift[-1]:.6f}, "
            f"global_drift_pressure={self.rs_global_drift_pressure[-1]:.4f}, "
            f"drift_alert_ratio={self.rs_drift_alert_ratio[-1]:.4f}, "
            f"clusters={int(self.rs_cluster_count[-1])}"
        )
