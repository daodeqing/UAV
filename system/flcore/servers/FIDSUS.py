import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from flcore.clients.clientFIDSUS import clientFIDSUS
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
        return F.cross_entropy(logits + adj.unsqueeze(0), y)


class FIDSUS(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_clients(clientFIDSUS)
        self.P = torch.diag(torch.ones(self.num_clients, device=self.device))
        self.uploaded_ids = []
        self.M = min(args.M, self.num_join_clients)
        self.client_models = [copy.deepcopy(self.global_model) for _ in range(self.num_clients)]
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []
        self.server_learning_rate = args.server_learning_rate
        self.head = self.client_models[0].head
        self.opt_h = torch.optim.SGD(self.head.parameters(), lr=self.server_learning_rate)
        self.head_train_epochs = int(getattr(args, "head_train_epochs", 2))
        self.head_criterion = LogitAdjustedCE(
            self.num_classes, tau=float(getattr(args, "la_tau", 1.0))
        ).to(self.device)

        self.rs_round_idx = []
        self.rs_eval_flag = []
        self.rs_force_affinity_flag = []

        self.rs_selected_clients = []
        self.rs_active_ratio = []
        self.rs_heavy_uploader_clients = []
        self.rs_heavy_uploader_ratio = []
        self.rs_upload_success_ratio = []
        self.rs_drop_count = []
        self.rs_bandwidth_budget_bytes = []
        self.rs_bandwidth_utilization = []

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

        self.rs_round_time = []
        self.rs_test_macro_f1 = []
        self.rs_test_weighted_f1 = []
        self.rs_test_macro_recall = []
        self.rs_test_balanced_acc = []
        self.rs_test_macro_auc = []
        self.rs_test_macro_prauc = []
        self.rs_active_clients = []
        self.rs_uploaded_clients = []
        self.rs_upload_ratio = []
        self.rs_proto_count = []
        self.rs_proto_upload_bytes = []
        self.rs_model_upload_bytes = []
        self.rs_total_upload_bytes = []
        self.rs_avg_client_train_time = []
        self.rs_avg_client_send_time = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()

            self.rs_round_idx.append(float(i))
            self.rs_eval_flag.append(float(1.0 if (i % self.eval_gap == 0) else 0.0))
            self.rs_force_affinity_flag.append(0.0)

            self.selected_clients = self.select_clients()
            self.send_models()
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate_personalized()
            for client in self.selected_clients:
                client.train()
            self.receive_models()
            self.train_head()
            self._sync_global_head()

            round_time = time.time() - s_t
            self.Budget.append(round_time)
            self.rs_round_time.append(float(round_time))
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])
            self._print_round_resource_metrics()

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        self.save_results()

    def _sync_global_head(self):
        for dst, src in zip(self.global_model.head.parameters(), self.head.parameters()):
            dst.data.copy_(src.data)

    def send_models(self):
        assert len(self.selected_clients) > 0
        for client in self.selected_clients:
            start_time = time.time()

            m_ = min(self.M, len(self.uploaded_ids))
            indices = torch.topk(self.P[client.id], m_).indices.tolist() if m_ > 0 else []
            send_ids = []
            send_models = []
            for idx in indices:
                send_ids.append(idx)
                send_models.append(self.client_models[idx])

            client.receive_models(send_ids, send_models)
            client.set_parameters(self.head)
            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert len(self.selected_clients) > 0

        active_clients = random.sample(
            self.selected_clients, int(self.client_activity_rate * self.current_num_join_clients)
        )

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_protos = []
        active_train_times = []
        active_send_times = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = (
                    client.train_time_cost["total_cost"] / client.train_time_cost["num_rounds"]
                    + client.send_time_cost["total_cost"] / client.send_time_cost["num_rounds"]
                )
            except ZeroDivisionError:
                client_time_cost = 0

            if client.train_time_cost["num_rounds"] > 0:
                active_train_times.append(client.train_time_cost["total_cost"] / client.train_time_cost["num_rounds"])
            if client.send_time_cost["num_rounds"] > 0:
                active_send_times.append(client.send_time_cost["total_cost"] / client.send_time_cost["num_rounds"])

            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                for cc in client.protos.keys():
                    y = torch.tensor(cc, dtype=torch.int64, device=self.device)
                    self.uploaded_protos.append((client.protos[cc], y))
                self.uploaded_weights.append(client.train_samples)
                self.client_models[client.id] = copy.deepcopy(client.model)
                self.P[client.id] += client.weight_vector
        if tot_samples > 0:
            for i, w in enumerate(self.uploaded_weights):
                self.uploaded_weights[i] = w / tot_samples

        proto_bytes = 0
        for p, _ in self.uploaded_protos:
            try:
                proto_bytes += int(p.numel() * p.element_size())
            except Exception:
                pass

        active_cnt = len(active_clients)
        uploaded_cnt = len(self.uploaded_ids)
        selected_cnt = len(getattr(self, "selected_clients", []))
        drop_cnt = max(active_cnt - uploaded_cnt, 0)
        self.rs_selected_clients.append(float(selected_cnt))
        self.rs_active_ratio.append(float(active_cnt / max(selected_cnt, 1)))
        self.rs_heavy_uploader_clients.append(float(uploaded_cnt))
        self.rs_heavy_uploader_ratio.append(float(uploaded_cnt / max(active_cnt, 1)))
        self.rs_upload_success_ratio.append(float(uploaded_cnt / max(active_cnt, 1)))
        self.rs_drop_count.append(float(drop_cnt))
        self.rs_bandwidth_budget_bytes.append(0.0)
        self.rs_bandwidth_utilization.append(0.0)

        self.rs_mode_full_ratio.append(1.0)
        self.rs_mode_freeze_base_ratio.append(0.0)
        self.rs_mode_head_only_ratio.append(0.0)
        self.rs_proto_drift_mean.append(0.0)
        self.rs_bid_loss_drop_mean.append(0.0)
        self.rs_bid_drift_mean.append(0.0)
        self.rs_bid_staleness_mean.append(0.0)
        self.rs_reject_time_count.append(float(drop_cnt))
        self.rs_reject_missing_delta_count.append(0.0)
        self.rs_reject_below_norm_count.append(0.0)
        self.rs_reject_total_count.append(float(drop_cnt))

        self.rs_active_clients.append(float(active_cnt))
        self.rs_uploaded_clients.append(float(uploaded_cnt))
        self.rs_upload_ratio.append(float(uploaded_cnt / max(active_cnt, 1)))
        self.rs_proto_count.append(float(len(self.uploaded_protos)))
        self.rs_proto_upload_bytes.append(float(proto_bytes))
        self.rs_model_upload_bytes.append(0.0)
        self.rs_total_upload_bytes.append(float(proto_bytes))
        self.rs_avg_client_train_time.append(float(np.mean(active_train_times)) if len(active_train_times) > 0 else 0.0)
        self.rs_avg_client_send_time.append(float(np.mean(active_send_times)) if len(active_send_times) > 0 else 0.0)

    def _print_round_resource_metrics(self):
        if len(self.rs_total_upload_bytes) == 0:
            return
        print(
            "[Round Metrics][FIDSUS] "
            f"selected={int(self.rs_selected_clients[-1])}, "
            f"active={int(self.rs_active_clients[-1])}, "
            f"uploaded={int(self.rs_uploaded_clients[-1])}, "
            f"upload_ratio={self.rs_upload_ratio[-1]:.4f}, "
            f"drop={int(self.rs_drop_count[-1])}, "
            f"proto_bytes={int(self.rs_proto_upload_bytes[-1])}, "
            f"total_upload_bytes={int(self.rs_total_upload_bytes[-1])}, "
            f"avg_train_time={self.rs_avg_client_train_time[-1]:.4f}s, "
            f"avg_send_time={self.rs_avg_client_send_time[-1]:.4f}s"
        )

    def train_head(self):
        if len(self.uploaded_protos) == 0:
            return
        ys = torch.stack([y for _, y in self.uploaded_protos]).view(-1).to(self.device)
        self.head_criterion.update_count(ys)
        proto_loader = DataLoader(self.uploaded_protos, self.batch_size, drop_last=False, shuffle=True)
        for _ in range(max(1, self.head_train_epochs)):
            for p, y in proto_loader:
                p = p.to(self.device)
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
        stats_train = self.train_metrics_personalized()
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        y_true_all = []
        y_prob_all = []
        for c in self.clients:
            y_true, y_prob = c.test_metrics_detail()
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)

        y_true = np.concatenate(y_true_all, axis=0)
        y_prob = np.concatenate(y_prob_all, axis=0)
        y_pred = np.argmax(y_prob, axis=1)
        y_true_bin = label_binarize(y_true, classes=np.arange(self.num_classes))

        test_acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        micro_auc = roc_auc_score(y_true_bin, y_prob, average="micro", multi_class="ovr")
        macro_auc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
        macro_prauc = average_precision_score(y_true_bin, y_prob, average="macro")

        if acc == None:
            self.rs_test_acc.append(float(test_acc))
        else:
            acc.append(float(test_acc))
        self.rs_test_auc.append(float(micro_auc))
        self.rs_test_macro_f1.append(float(macro_f1))
        self.rs_test_weighted_f1.append(float(weighted_f1))
        self.rs_test_macro_recall.append(float(macro_recall))
        self.rs_test_balanced_acc.append(float(balanced_acc))
        self.rs_test_macro_auc.append(float(macro_auc))
        self.rs_test_macro_prauc.append(float(macro_prauc))
        if loss == None:
            self.rs_train_loss.append(float(train_loss))
        else:
            loss.append(float(train_loss))
        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Macro-F1: {:.4f}".format(macro_f1))
        print("Weighted-F1: {:.4f}".format(weighted_f1))
        print("Macro-Recall: {:.4f}".format(macro_recall))
        print("Balanced Accuracy: {:.4f}".format(balanced_acc))
        print("Micro AUC: {:.4f}".format(micro_auc))
        print("Macro AUC: {:.4f}".format(macro_auc))
        print("Macro PR-AUC: {:.4f}".format(macro_prauc))
