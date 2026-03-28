import copy
import time
from collections import defaultdict

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


class clientFIDSUS(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.num_clients = args.num_clients
        self.old_model = copy.deepcopy(self.model)
        self.received_ids = []
        self.received_models = []
        self.weight_vector = torch.zeros(self.num_clients, device=self.device)
        self.val_ratio = 0.2
        self.train_samples = int(self.train_samples * (1 - self.val_ratio))
        self.batch_size = args.batch_size
        self.mu = args.mu
        self.model_per = copy.deepcopy(self.model)
        self.optimizer_per = PerturbedGradientDescent(
            self.model_per.parameters(), lr=self.learning_rate, mu=self.mu
        )
        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per, gamma=args.learning_rate_decay_gamma
        )
        self.CEloss = nn.CrossEntropyLoss()
        self.head_per = self.model_per.head
        self.opt_h_per = torch.optim.SGD(self.head_per.parameters(), lr=self.learning_rate)

    def train(self):
        trainloader, val_loader = self.load_train_data()
        start_time = time.time()
        self.aggregate_parameters(val_loader)
        self.clone_model(self.model, self.old_model)
        shared_ref = [p.detach().clone() for p in self.model.parameters()]

        self.model.train()
        self.model_per.train()
        protos = defaultdict(list)
        protos_per = defaultdict(list)

        for _ in range(self.local_epochs):
            for x, y in trainloader:
                x, y = self._to_device(x, y)

                reg = self.model.base(x)
                output = self.model.head(reg)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                reg_per = self.model_per.base(x)
                output_per = self.model_per.head(reg_per)
                loss_per = self.loss(output_per, y)
                self.optimizer_per.zero_grad()
                loss_per.backward()
                self.optimizer_per.step(shared_ref, self.device)

                for i, yy in enumerate(y):
                    label = int(yy.item())
                    protos[label].append(reg[i, :].detach().data)
                    protos_per[label].append(reg_per[i, :].detach().data)

        self.protos_g = agg_func(protos)
        self.protos_per = agg_func(protos_per)
        self.protos = aggregation(self.protos_g, self.protos_per)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_per.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def load_train_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        train_data = read_client_data_un(self.dataset, self.id, is_train=True)
        if len(train_data) < 5:
            loader = DataLoader(train_data, batch_size, drop_last=False, shuffle=False)
            return loader, loader
        val_num = max(1, int(self.val_ratio * len(train_data)))
        val_data = train_data[-val_num:]
        train_data = train_data[:-val_num]
        trainloader = DataLoader(train_data, batch_size, drop_last=False, shuffle=True)
        val_loader = DataLoader(val_data, batch_size, drop_last=False, shuffle=False)
        return trainloader, val_loader

    def receive_models(self, ids, models):
        self.received_ids = ids
        self.received_models = models

    def set_parameters(self, head):
        for new_param, old_param in zip(head.parameters(), self.model.head.parameters()):
            old_param.data = new_param.data.clone()
        for new_param, old_param in zip(head.parameters(), self.model_per.head.parameters()):
            old_param.data = new_param.data.clone()

    def _build_candidate(self, recv_model):
        cand = copy.deepcopy(self.model)
        for p_c, p_r in zip(cand.base.parameters(), recv_model.base.parameters()):
            p_c.data.copy_(p_r.data)
        for p_c, p_h in zip(cand.head.parameters(), self.model.head.parameters()):
            p_c.data.copy_(p_h.data)
        return cand

    def weight_cal(self, val_loader):
        if len(self.received_models) == 0:
            self.weight_vector = torch.zeros(self.num_clients, device=self.device)
            return torch.tensor([], device=self.device)

        weight_list = []
        baseline = self.recalculate_loss(self.model, val_loader)
        for recv_model in self.received_models:
            cand = self._build_candidate(recv_model)
            gain = baseline - self.recalculate_loss(cand, val_loader)
            base_diff = []
            for p_r, p_l in zip(recv_model.base.parameters(), self.model.base.parameters()):
                base_diff.append((p_r.data - p_l.data).reshape(-1))
            base_diff = torch.cat(base_diff)
            weight_list.append(gain / (torch.norm(base_diff) + 1e-6))
        self.weight_vector_update(weight_list)
        return torch.stack(weight_list)

    def weight_vector_update(self, weight_list):
        vec = np.zeros(self.num_clients, dtype=np.float32)
        for w, cid in zip(weight_list, self.received_ids):
            vec[cid] += float(w.item())
        self.weight_vector = torch.tensor(vec, device=self.device)

    def recalculate_loss(self, new_model, val_loader):
        total = 0.0
        count = 0
        new_model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = self._to_device(x, y)
                output = new_model(x)
                total += float(self.loss(output, y).item())
                count += 1
        new_model.train()
        return total / max(count, 1)

    def aggregate_parameters(self, val_loader):
        weights = self.weight_scale(self.weight_cal(val_loader))
        if len(weights) == 0:
            return

        old_base = [p.data.clone() for p in self.model.base.parameters()]
        new_base = [p.clone() for p in old_base]

        for w, recv_model in zip(weights, self.received_models):
            for nb, ob, rp in zip(new_base, old_base, recv_model.base.parameters()):
                nb.add_((rp.data - ob) * float(w))

        for p, nb in zip(self.model.base.parameters(), new_base):
            p.data.copy_(nb)

    def weight_scale(self, weights):
        if weights.numel() == 0:
            return torch.tensor([], device=self.device)
        weights = torch.clamp_min(weights, 0.0)
        w_sum = torch.sum(weights)
        if float(w_sum.item()) > 0:
            return weights / w_sum
        return torch.tensor([], device=self.device)

    def _to_device(self, x, y):
        if isinstance(x, list):
            x[0] = x[0].to(self.device)
            x = x[0]
        else:
            x = x.to(self.device)
        y = y.to(self.device)
        return x, y

    def train_metrics_personalized(self):
        trainloader, _ = self.load_train_data()
        self.model_per.eval()
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                x, y = self._to_device(x, y)
                output = self.model_per(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        return losses, train_num

    def train_metrics(self):
        trainloader, _ = self.load_train_data()
        self.model.eval()
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                x, y = self._to_device(x, y)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
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
            for x, y in testloaderfull:
                x, y = self._to_device(x, y)
                output = self.model_per(x)
                y_true.append(y.detach().cpu().numpy())
                y_prob.append(F.softmax(output, dim=1).detach().cpu().numpy())
        y_true = np.concatenate(y_true, axis=0)
        y_prob = np.concatenate(y_prob, axis=0)
        return y_true, y_prob


def agg_func(protos):
    for label, proto_list in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for item in proto_list:
                proto += item.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]
    return protos


def proto_mix_weight(cur, old, tau=8.0, w_min=0.15, w_max=0.85):
    cur = F.normalize(cur, dim=0)
    old = F.normalize(old, dim=0)
    dist = torch.mean((cur - old) ** 2)
    return torch.exp(-tau * dist).clamp(w_min, w_max)


def aggregation(protos, protos_per):
    out = {}
    all_labels = set(protos.keys()) | set(protos_per.keys())
    for label in all_labels:
        if label in protos and label in protos_per:
            w = proto_mix_weight(protos[label], protos_per[label])
            out[label] = w * protos[label] + (1.0 - w) * protos_per[label]
        elif label in protos:
            out[label] = protos[label]
        else:
            out[label] = protos_per[label]
    return out
