import argparse
import csv
import copy
import datetime
import json
import logging
import os
import random
import re
import sys
import time
import warnings

import numpy as np
import torch

from flcore.servers.FIDSUS import FIDSUS
from flcore.servers.serverHGFIDSUS import HGFIDSUS
from flcore.servers.serverSGEFIDSUS import SGEFIDSUS
from flcore.servers.serverSkyGuardPFIDS import SkyGuardPFIDS
from flcore.servers.serveravg import FedAvg
from flcore.servers.serveravgDBE import FedAvgDBE
from flcore.servers.servergh import FedGH
from flcore.servers.servergpfl import GPFL
from flcore.servers.servermoon import MOON
from flcore.servers.serverproto import FedProto
from flcore.servers.serverprox import FedProx
from flcore.trainmodel.models import *
from utils.result_utils import average_data

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")

emb_dim = 32
torch.manual_seed(10)


class _RoundMetricTee:
    """Tee stdout to file and extract round/loss/acc records into CSV."""

    _round_re = re.compile(r"Round number:\s*(\d+)")
    _loss_re = re.compile(r"^Averaged Train Loss:\s*([+-]?\d+(?:\.\d+)?)")
    _acc_re = re.compile(r"^Averaged Test Accuracy:\s*([+-]?\d+(?:\.\d+)?)")
    _macro_f1_re = re.compile(r"^Macro-F1:\s*([+-]?\d+(?:\.\d+)?)")
    _weighted_f1_re = re.compile(r"^Weighted-F1:\s*([+-]?\d+(?:\.\d+)?)")
    _balanced_acc_re = re.compile(r"^Balanced Accuracy:\s*([+-]?\d+(?:\.\d+)?)")
    _macro_prauc_re = re.compile(r"^Macro PR-AUC:\s*([+-]?\d+(?:\.\d+)?)")
    _ece_re = re.compile(r"^ECE:\s*([+-]?\d+(?:\.\d+)?)")
    _brier_re = re.compile(r"^Brier Score:\s*([+-]?\d+(?:\.\d+)?)")
    _avg_conf_re = re.compile(r"^Average Confidence:\s*([+-]?\d+(?:\.\d+)?)")
    _few_shot_recall_re = re.compile(r"^Few-shot Recall:\s*([+-]?\d+(?:\.\d+)?)")
    _few_shot_macro_f1_re = re.compile(r"^Few-shot Macro-F1:\s*([+-]?\d+(?:\.\d+)?)")
    _ema_acc_re = re.compile(r"^EMA Test Accuracy:\s*([+-]?\d+(?:\.\d+)?)")
    _selection_score_re = re.compile(r"^Selection Score \([^)]+\):\s*([+-]?\d+(?:\.\d+)?)")
    _round_metrics_re = re.compile(r"^\[Round Metrics\]")

    def __init__(self, run_log_path: str, round_csv_path: str, mirror_stream):
        self.run_log_path = run_log_path
        self.round_csv_path = round_csv_path
        self._mirror = mirror_stream
        self._log_fp = open(run_log_path, "w", encoding="utf-8", buffering=1)
        self._csv_fp = open(round_csv_path, "w", encoding="utf-8", newline="")
        self._csv_writer = csv.writer(self._csv_fp)
        self._csv_writer.writerow(
            [
                "timestamp",
                "round",
                "train_loss",
                "test_acc",
                "macro_f1",
                "weighted_f1",
                "balanced_acc",
                "macro_prauc",
                "ece",
                "brier",
                "avg_conf",
                "few_shot_recall",
                "few_shot_macro_f1",
                "ema_test_acc",
                "selection_score",
            ]
        )
        self._buf = ""
        self._current_round = None
        self._pending_loss = None
        self._pending_eval = {}

    def write(self, data):
        if not isinstance(data, str):
            data = str(data)

        self._mirror.write(data)
        self._mirror.flush()

        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._handle_line(line)

    def flush(self):
        self._mirror.flush()
        self._log_fp.flush()
        self._csv_fp.flush()

    def close(self):
        if self._buf:
            self._handle_line(self._buf)
            self._buf = ""
        self._log_fp.flush()
        self._csv_fp.flush()
        self._log_fp.close()
        self._csv_fp.close()

    def _handle_line(self, line: str):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log_fp.write(f"[{ts}] {line}\n")

        m_round = self._round_re.search(line)
        if m_round:
            self._current_round = int(m_round.group(1))
            self._pending_loss = None
            self._pending_eval = {}

        m_loss = self._loss_re.search(line)
        if m_loss:
            try:
                self._pending_loss = float(m_loss.group(1))
            except Exception:
                self._pending_loss = None

        m_acc = self._acc_re.search(line)
        if m_acc:
            try:
                self._pending_eval["test_acc"] = float(m_acc.group(1))
            except Exception:
                pass

        for key, pattern in [
            ("macro_f1", self._macro_f1_re),
            ("weighted_f1", self._weighted_f1_re),
            ("balanced_acc", self._balanced_acc_re),
            ("macro_prauc", self._macro_prauc_re),
            ("ece", self._ece_re),
            ("brier", self._brier_re),
            ("avg_conf", self._avg_conf_re),
            ("few_shot_recall", self._few_shot_recall_re),
            ("few_shot_macro_f1", self._few_shot_macro_f1_re),
        ]:
            m_val = pattern.search(line)
            if m_val:
                try:
                    self._pending_eval[key] = float(m_val.group(1))
                except Exception:
                    pass

        m_ema = self._ema_acc_re.search(line)
        if m_ema:
            try:
                self._pending_eval["ema_test_acc"] = float(m_ema.group(1))
            except Exception:
                return
            return

        m_selection = self._selection_score_re.search(line)
        if m_selection:
            try:
                self._pending_eval["selection_score"] = float(m_selection.group(1))
            except Exception:
                pass
            self._flush_eval_row(ts)
            return

        if self._round_metrics_re.search(line) and ("test_acc" in self._pending_eval):
            self._flush_eval_row(ts)

    def _flush_eval_row(self, ts: str):
        self._csv_writer.writerow([
            ts,
            "" if self._current_round is None else int(self._current_round),
            "" if self._pending_loss is None else float(self._pending_loss),
            self._pending_eval.get("test_acc", ""),
            self._pending_eval.get("macro_f1", ""),
            self._pending_eval.get("weighted_f1", ""),
            self._pending_eval.get("balanced_acc", ""),
            self._pending_eval.get("macro_prauc", ""),
            self._pending_eval.get("ece", ""),
            self._pending_eval.get("brier", ""),
            self._pending_eval.get("avg_conf", ""),
            self._pending_eval.get("few_shot_recall", ""),
            self._pending_eval.get("few_shot_macro_f1", ""),
            self._pending_eval.get("ema_test_acc", ""),
            self._pending_eval.get("selection_score", ""),
        ])
        self._csv_fp.flush()
        self._pending_eval = {}


def _set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _auto_configure_dataset_args(args):
    try:
        base_dir = os.path.dirname(__file__)
        cfg_path = os.path.join(base_dir, "..", "dataset", args.dataset, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if getattr(args, "auto_num_classes", True) and "num_classes" in cfg:
                args.num_classes = int(cfg["num_classes"])
            if getattr(args, "auto_num_clients", True) and "num_clients" in cfg:
                args.num_clients = int(cfg["num_clients"])
    except Exception:
        return


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def run(args):
    time_list = []
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        if model_str == "1dcnn":
            args.model = CNN1D(hidden_dim=emb_dim, num_classes=args.num_classes).to(args.device)
        elif model_str == "tcn":
            args.model = TCN1D(hidden_dim=emb_dim, num_layers=3, num_classes=args.num_classes).to(args.device)
        else:
            raise NotImplementedError

        print(args.model)

        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)
        elif args.algorithm == "FedProx":
            server = FedProx(args, i)
        elif args.algorithm == "FedProto":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedProto(args, i)
        elif args.algorithm == "MOON":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = MOON(args, i)
        elif args.algorithm == "GPFL":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = GPFL(args, i)
        elif args.algorithm == "FedGH":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGH(args, i)
        elif args.algorithm == "FedAvgDBE":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvgDBE(args, i)
        elif args.algorithm == "FIDSUS":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FIDSUS(args, i)
        elif args.algorithm == "HGFIDSUS":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = HGFIDSUS(args, i)
        elif args.algorithm in {"SGEFIDSUS", "PSGEFIDSUS"}:
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = SGEFIDSUS(args, i)
        elif args.algorithm == "SkyGuardPFIDS":
            args.head = copy.deepcopy(args.model.fc)
            feat_dim = int(args.head.in_features)
            args.model.fc = nn.Identity()
            adapter = ResidualAdapter(
                dim=feat_dim,
                bottleneck_dim=args.skyguard_adapter_dim,
                dropout=args.skyguard_adapter_dropout,
            ).to(args.device)
            args.model = BaseHeadSplit(args.model, args.head, adapter=adapter).to(args.device)
            server = SkyGuardPFIDS(args, i)
        else:
            raise NotImplementedError

        server.train()
        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    savetime_dir = "timecost"
    os.makedirs(savetime_dir, exist_ok=True)
    timefilename = os.path.join(
        savetime_dir, f"time_cost_{args.algorithm}_{args.num_clients}_{args.dataset}_{args.goal}.txt"
    )
    with open(timefilename, "w") as file:
        file.write(f"Total time cost: {round(np.average(time_list), 2)}s.\n")

    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)
    print("All done!")


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("-go", "--goal", type=str, default="test", help="The goal for this experiment")
    parser.add_argument("-dev", "--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("-did", "--device_id", type=str, default="0")
    parser.add_argument("-data", "--dataset", type=str, default="FANET")
    parser.add_argument("-nc", "--num_clients", type=int, default=50, help="Total number of clients")
    parser.add_argument("-jr", "--join_ratio", type=float, default=0.4, help="Ratio of clients per round")
    parser.add_argument("-car", "--client_activity_rate", type=float, default=0.8)
    parser.add_argument("-M", "--M", type=int, default=5, help="Server only sends M client models to one client at each round")
    parser.add_argument("-nb", "--num_classes", type=int, default=10)
    parser.add_argument("-m", "--model", type=str, default="tcn")
    parser.add_argument("-lbs", "--batch_size", type=int, default=64)
    parser.add_argument("-lr", "--local_learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("-slr", "--server_learning_rate", type=float, default=0.01)
    parser.add_argument("-ld", "--learning_rate_decay", type=_str2bool, default=False)
    parser.add_argument("-ldg", "--learning_rate_decay_gamma", type=float, default=0.98)
    parser.add_argument("-gr", "--global_rounds", type=int, default=100)
    parser.add_argument("-ls", "--local_epochs", type=int, default=1, help="Multiple update steps in one local epoch.")
    parser.add_argument("-algo", "--algorithm", type=str, default="SkyGuardPFIDS")
    parser.add_argument("-rjr", "--random_join_ratio", type=_str2bool, default=False, help="Random ratio of clients per round")
    parser.add_argument("-pv", "--prev", type=int, default=0, help="Previous Running times")
    parser.add_argument("-t", "--times", type=int, default=1, help="Running times")
    parser.add_argument("-eg", "--eval_gap", type=int, default=5, help="Rounds gap for evaluation")
    parser.add_argument("-sfn", "--save_folder_name", type=str, default="items")
    parser.add_argument("-bnpc", "--batch_num_per_client", type=int, default=2)
    parser.add_argument("-tth", "--time_threthold", type=float, default=10000, help="The threthold for droping slow clients")
    parser.add_argument("-bt", "--beta", type=float, default=0.0)
    parser.add_argument("-lam", "--lamda", type=float, default=0.99, help="Regularization weight")
    parser.add_argument("-mu", "--mu", type=float, default=0.01)
    parser.add_argument("-lrp", "--p_learning_rate", type=float, default=0.01, help="personalized learning rate")
    parser.add_argument("-tau", "--tau", type=float, default=1.0)
    parser.add_argument("-mo", "--momentum", type=float, default=0.1)
    parser.add_argument("-klw", "--kl_weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=10, help="Global random seed for reproducibility")

    # HGFIDSUS specific
    parser.add_argument("--auto_num_classes", type=_str2bool, default=True)
    parser.add_argument("--auto_num_clients", type=_str2bool, default=True)
    parser.add_argument("--client_partition", type=str, default="by_session")
    parser.add_argument("--dir_alpha", type=float, default=0.5)
    parser.add_argument("--public_val_ratio", type=float, default=0.05)
    parser.add_argument("--public_val_batches", type=int, default=2)
    parser.add_argument("--poison_ratio", type=float, default=0.0)
    parser.add_argument("--sybil_ratio", type=float, default=0.0)
    parser.add_argument("--dropout_ratio", type=float, default=0.0)
    parser.add_argument("--energy_budget", type=float, default=10.0)
    parser.add_argument("--payment_budget", type=float, default=0.0)
    parser.add_argument("--price_compute", type=float, default=1.0)
    parser.add_argument("--price_comm", type=float, default=1.0)
    parser.add_argument("--dynamic_pricing", type=_str2bool, default=True)
    parser.add_argument("--price_step", type=float, default=0.03)
    parser.add_argument("--price_compute_min", type=float, default=0.7)
    parser.add_argument("--price_compute_max", type=float, default=1.40)
    parser.add_argument("--price_comm_min", type=float, default=0.7)
    parser.add_argument("--price_comm_max", type=float, default=2.0)
    parser.add_argument("--price_time_tolerance", type=float, default=0.05)
    parser.add_argument("--price_target_round_time", type=float, default=0.0)
    parser.add_argument("--price_warmup_rounds", type=int, default=5)
    parser.add_argument("--price_acc_drop_th", type=float, default=0.01)
    parser.add_argument("--price_acc_gain_th", type=float, default=0.01)
    parser.add_argument("--price_staleness_target", type=float, default=3.0)
    parser.add_argument("--price_entropy_relax_th", type=float, default=0.60)
    parser.add_argument("--bandwidth_budget", type=int, default=0)
    parser.add_argument("--comm_budget_mb", type=float, default=12.0)
    parser.add_argument("--challenge_batches", type=int, default=2)
    parser.add_argument("--server_verify_topk", type=int, default=10)
    parser.add_argument("--auction_winners_frac", type=float, default=0.30)
    parser.add_argument("--delta_min_norm", type=float, default=1e-12)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--affinity_drift_th", type=float, default=0.08)
    parser.add_argument("--affinity_min_prob", type=float, default=0.35)
    parser.add_argument("--force_affinity_drop_rounds", type=int, default=3)
    parser.add_argument("--force_affinity_drop_threshold", type=float, default=0.03)
    parser.add_argument("--rep_decay", type=float, default=0.995)
    parser.add_argument("--rep_bonus", type=float, default=0.01)
    parser.add_argument("--rep_penalty", type=float, default=0.01)
    parser.add_argument("--bid_w1", type=float, default=1.0)
    parser.add_argument("--bid_w2", type=float, default=0.7)
    parser.add_argument("--bid_w3", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.01)
    parser.add_argument("--grad_clip_norm", type=float, default=3.0)
    parser.add_argument("--peer_mixing", type=float, default=0.30)
    parser.add_argument("--personalized_sync_alpha", type=float, default=0.5)
    parser.add_argument("--delta_server_lr", type=float, default=0.05)
    parser.add_argument("--warmup_rounds", type=int, default=8)
    parser.add_argument("--global_base_alpha", type=float, default=0.10)
    parser.add_argument("--head_train_epochs", type=int, default=2)
    parser.add_argument("--la_tau", type=float, default=1.0)
    parser.add_argument("--delta_clip_norm", type=float, default=3.0)
    parser.add_argument("--delta_trim_ratio", type=float, default=0.1)
    parser.add_argument("--max_staleness_upload", type=int, default=0)
    parser.add_argument("--stale_priority_th", type=int, default=4)
    parser.add_argument("--stale_quota_frac", type=float, default=0.3)
    parser.add_argument("--stale_score_alpha", type=float, default=0.25)
    parser.add_argument("--stale_score_cap", type=int, default=30)
    parser.add_argument("--eval_acc_ema_alpha", type=float, default=0.2)

    # Latest HG update knobs
    parser.add_argument("--loader_num_workers", type=int, default=2)
    parser.add_argument("--cache_client_data", type=_str2bool, default=True)
    parser.add_argument(
        "--eval_loader_num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers used by evaluation/validation/calibration loaders.",
    )
    parser.add_argument(
        "--eval_pin_memory",
        type=_str2bool,
        default=False,
        help="Whether to enable pin_memory for evaluation/validation/calibration loaders.",
    )
    parser.add_argument("--affinity_eval_batches", type=int, default=2)
    parser.add_argument("--bid_eval_batches", type=int, default=2)
    parser.add_argument("--full_train_rounds", type=int, default=35)
    parser.add_argument("--freeze_base_price", type=float, default=1.65)
    parser.add_argument("--head_only_price", type=float, default=2.30)
    parser.add_argument("--entropy_keep_full_th", type=float, default=0.55)
    parser.add_argument("--drift_keep_full_th", type=float, default=0.08)
    parser.add_argument("--stale_force_full_th", type=int, default=12)
    parser.add_argument("--compute_cost_weight", type=float, default=0.20)
    parser.add_argument("--comm_cost_weight", type=float, default=1.00)
    parser.add_argument("--uncertainty_bonus_weight", type=float, default=0.20)
    parser.add_argument("--min_delta_uploaders", type=int, default=6)
    parser.add_argument("--use_failover", type=_str2bool, default=False)
    parser.add_argument("--failover_decay", type=float, default=0.75)
    parser.add_argument("--failover_max_age", type=int, default=8)
    parser.add_argument("--early_stop_patience", type=int, default=6)
    parser.add_argument("--early_stop_min_rounds", type=int, default=30)
    parser.add_argument("--checkpoint_mode", type=str, default="multi", choices=["acc", "multi"])
    parser.add_argument("--checkpoint_w_acc", type=float, default=0.35)
    parser.add_argument("--checkpoint_w_macro_f1", type=float, default=0.25)
    parser.add_argument("--checkpoint_w_balanced_acc", type=float, default=0.20)
    parser.add_argument("--checkpoint_w_few_shot_macro_f1", type=float, default=0.10)
    parser.add_argument("--checkpoint_w_ece", type=float, default=0.05)
    parser.add_argument("--checkpoint_w_brier", type=float, default=0.05)
    parser.add_argument("--enable_contracts", type=_str2bool, default=False)
    parser.add_argument("--contract_start_round", type=int, default=8)
    parser.add_argument("--min_full_clients", type=int, default=6)
    parser.add_argument("--contract_compute_discount", type=float, default=0.25)
    parser.add_argument("--contract_comm_discount", type=float, default=0.20)
    parser.add_argument("--contract_bonus", type=float, default=0.10)
    parser.add_argument("--enable_sticky_sampling", type=_str2bool, default=False)
    parser.add_argument("--sticky_fraction", type=float, default=0.50)
    parser.add_argument("--no_replace_window", type=int, default=3)
    parser.add_argument("--enable_delta_compression", type=_str2bool, default=True)
    parser.add_argument("--delta_topk", type=float, default=0.25)
    parser.add_argument("--delta_error_feedback", type=_str2bool, default=True)
    parser.add_argument("--delta_topk_warmup_rounds", type=int, default=30)
    parser.add_argument("--delta_topk_warmup_value", type=float, default=0.60)
    parser.add_argument("--contract_force_dense_delta", type=_str2bool, default=True)
    parser.add_argument("--delta_tail_boost", type=float, default=0.0)
    parser.add_argument("--delta_cred_boost", type=float, default=0.0)
    parser.add_argument("--delta_adaptive_topk_max", type=float, default=0.80)
    parser.add_argument("--proto_fp16", type=_str2bool, default=True)
    parser.add_argument("--staleness_decay_gamma", type=float, default=0.10)
    parser.add_argument("--eval_calibration", type=_str2bool, default=True)
    parser.add_argument("--eval_calibration_start_round", type=int, default=0)
    parser.add_argument("--eval_calibration_temp_min", type=float, default=0.6)
    parser.add_argument("--eval_calibration_temp_max", type=float, default=3.0)
    parser.add_argument("--eval_calibration_temp_steps", type=int, default=25)
    parser.add_argument(
        "--eval_calibration_max_batches",
        type=int,
        default=0,
        help="Maximum number of validation batches per client used for calibration; 0 means full validation.",
    )
    parser.add_argument(
        "--eval_train_max_batches",
        type=int,
        default=2,
        help="Maximum number of train batches per client used when computing averaged train loss during evaluation; 0 means full train set.",
    )
    parser.add_argument(
        "--eval_test_max_batches",
        type=int,
        default=0,
        help="Maximum number of test batches per client used for evaluation metrics; 0 means full test set.",
    )
    parser.add_argument(
        "--eval_progress_log",
        type=_str2bool,
        default=True,
        help="Whether to print per-client progress during personalized evaluation.",
    )

    # SGEFIDSUS specific
    parser.add_argument("--u_gain", type=float, default=1.0)
    parser.add_argument("--u_tail", type=float, default=0.25)
    parser.add_argument("--u_uncertainty", type=float, default=0.2)
    parser.add_argument("--u_rep", type=float, default=0.15)
    parser.add_argument("--u_few_shot", type=float, default=0.20)
    parser.add_argument("--sgef_enable_contracts", type=_str2bool, default=False)
    parser.add_argument("--sgef_enable_sticky_sampling", type=_str2bool, default=False)
    parser.add_argument("--sgef_use_failover", type=_str2bool, default=False)
    parser.add_argument("--prototype_logit_weight", type=float, default=0.60)
    parser.add_argument("--prototype_loss_weight", type=float, default=0.25)
    parser.add_argument("--prototype_temperature", type=float, default=1.0)
    parser.add_argument("--prototype_memory_momentum", type=float, default=0.80)
    parser.add_argument("--prototype_server_momentum", type=float, default=0.20)
    parser.add_argument("--prototype_global_mix", type=float, default=0.55)
    parser.add_argument("--prototype_l2_norm", type=_str2bool, default=True)
    parser.add_argument("--few_shot_threshold", type=int, default=20)
    parser.add_argument("--few_shot_loss_boost", type=float, default=0.50)
    parser.add_argument("--few_shot_proto_boost", type=float, default=0.40)
    parser.add_argument("--global_few_shot_threshold", type=float, default=100.0)
    parser.add_argument("--few_shot_selection_bonus", type=float, default=0.20)

    # SkyGuardPFIDS specific
    parser.add_argument("--skyguard_num_clusters", type=int, default=3)
    parser.add_argument("--skyguard_adapter_dim", type=int, default=16)
    parser.add_argument("--skyguard_adapter_dropout", type=float, default=0.10)
    parser.add_argument("--skyguard_lambda_f", type=float, default=0.35)
    parser.add_argument("--skyguard_lambda_p", type=float, default=0.20)
    parser.add_argument("--skyguard_focal_gamma", type=float, default=2.0)
    parser.add_argument("--skyguard_w_gain", type=float, default=0.60)
    parser.add_argument("--skyguard_w_shapley", type=float, default=0.60)
    parser.add_argument("--skyguard_w_grad", type=float, default=0.30)
    parser.add_argument("--skyguard_w_quality", type=float, default=0.20)
    parser.add_argument("--skyguard_w_resource", type=float, default=0.35)
    parser.add_argument("--skyguard_w_attack", type=float, default=0.45)
    parser.add_argument("--skyguard_replicator_lr", type=float, default=0.10)
    parser.add_argument("--skyguard_uncertainty_bonus", type=float, default=0.05)
    parser.add_argument("--skyguard_prob_floor", type=float, default=0.10)
    parser.add_argument("--skyguard_prob_ceiling", type=float, default=0.90)
    parser.add_argument("--skyguard_online_floor", type=float, default=0.20)
    parser.add_argument("--skyguard_profile_momentum", type=float, default=0.80)
    parser.add_argument("--skyguard_adapter_momentum", type=float, default=0.70)
    parser.add_argument("--skyguard_entropy_drift_th", type=float, default=0.70)
    parser.add_argument("--skyguard_proto_drift_th", type=float, default=0.08)
    parser.add_argument("--skyguard_personalization_boost", type=float, default=0.25)
    parser.add_argument("--skyguard_scheduler_prior_mix", type=float, default=0.10)
    parser.add_argument("--skyguard_sybil_similarity_th", type=float, default=0.85)
    parser.add_argument("--skyguard_robust_min_factor", type=float, default=0.10)
    parser.add_argument("--skyguard_require_public_val", type=_str2bool, default=True)
    parser.add_argument("--skyguard_require_real_profiles", type=_str2bool, default=True)

    args = parser.parse_args()

    _set_global_seed(int(args.seed))
    _auto_configure_dataset_args(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Goal: {}".format(args.goal))
    print("Local epochs: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client activity rate: {}".format(args.client_activity_rate))
    print("Seed: {}".format(args.seed))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("=" * 50)

    run_ts = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    run_tag = f"{args.algorithm}_{args.dataset}_{args.goal}_{run_ts}"
    run_log_path = os.path.join(log_dir, f"run_{run_tag}.log")
    round_csv_path = os.path.join(log_dir, f"round_metrics_{run_tag}.csv")

    _orig_stdout = sys.stdout
    _tee = _RoundMetricTee(run_log_path=run_log_path, round_csv_path=round_csv_path, mirror_stream=_orig_stdout)
    sys.stdout = _tee

    try:
        print(f"Run log: {run_log_path}")
        print(f"Round metrics CSV: {round_csv_path}")
        run(args)
    finally:
        sys.stdout = _orig_stdout
        _tee.close()
        print(f"Saved run log to: {run_log_path}")
        print(f"Saved round metrics to: {round_csv_path}")
