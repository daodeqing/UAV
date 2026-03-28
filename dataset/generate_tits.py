import argparse
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils.dataset_utils import check, save_file, separate_data, split_data


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _preprocess_tabular(csv_path: str, label_col: str):
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"label column '{label_col}' not found in {csv_path}")

    y_raw = df[label_col].astype(str).values
    X_df = df.drop(columns=[label_col]).copy()

    for c in X_df.columns:
        if X_df[c].dtype == object:
            X_df[c] = X_df[c].astype("category").cat.codes
    X_df = X_df.fillna(0)

    X = X_df.values.astype(np.float32)
    y = LabelEncoder().fit_transform(y_raw)
    num_classes = int(len(np.unique(y)))

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    return X, y, num_classes


def _normalize_histogram(statistic, num_classes: int):
    hist = np.zeros((len(statistic), num_classes), dtype=np.float32)
    totals = np.zeros((len(statistic),), dtype=np.float32)
    for cid, pairs in enumerate(statistic):
        for label, count in pairs:
            label_int = int(label)
            count_float = float(count)
            if 0 <= label_int < num_classes:
                hist[cid, label_int] = count_float
                totals[cid] += count_float
    norm = hist / np.clip(totals[:, None], 1.0, None)
    return hist, norm, totals


def _build_uav_context(statistic, num_classes: int, num_clusters: int):
    hist, norm_hist, totals = _normalize_histogram(statistic, num_classes)
    num_clients = len(statistic)
    max_total = float(max(float(totals.max()) if totals.size > 0 else 1.0, 1.0))
    ent = -np.sum(norm_hist * np.log(np.clip(norm_hist, 1e-12, 1.0)), axis=1)
    ent = ent / max(np.log(max(num_classes, 2)), 1e-8)
    dominant = norm_hist.max(axis=1) if norm_hist.size > 0 else np.zeros((num_clients,), dtype=np.float32)
    support = totals / max_total

    cluster_count = int(max(1, min(num_clusters, num_clients)))
    if num_clients >= cluster_count and cluster_count > 1:
        feats = np.concatenate([norm_hist, support[:, None], dominant[:, None], ent[:, None]], axis=1)
        clusters = KMeans(n_clusters=cluster_count, random_state=0, n_init=10).fit_predict(feats).tolist()
    else:
        clusters = [int(i % cluster_count) for i in range(num_clients)]

    profiles = []
    for cid in range(num_clients):
        phase = 0.5 * (1.0 + np.sin(0.61 * (cid + 1)))
        compute_capacity = float(np.clip(0.35 + 0.40 * support[cid] + 0.25 * dominant[cid], 0.10, 0.99))
        link_quality = float(np.clip(0.30 + 0.35 * (1.0 - ent[cid]) + 0.35 * phase, 0.10, 0.99))
        energy_level = float(np.clip(0.30 + 0.35 * dominant[cid] + 0.35 * (1.0 - phase), 0.10, 0.99))
        expected_uptime = float(
            np.clip(0.15 + 0.30 * compute_capacity + 0.30 * link_quality + 0.25 * energy_level, 0.10, 0.99)
        )
        profiles.append(
            {
                "client_id": int(cid),
                "cluster_id": int(clusters[cid]),
                "compute_capacity": compute_capacity,
                "link_quality": link_quality,
                "energy_level": energy_level,
                "expected_uptime": expected_uptime,
                "dropout_risk": float(np.clip(1.0 - expected_uptime, 0.01, 0.95)),
                "label_entropy": float(ent[cid]),
                "dominant_share": float(dominant[cid]),
                "support_ratio": float(support[cid]),
            }
        )

    return {
        "client_clusters": [int(c) for c in clusters],
        "client_profiles": profiles,
        "profile_source": "synthetic_from_partition_statistics",
    }


def generate_tits(
    csv_path: str,
    output_dir: str,
    label_col: str,
    num_clients: int,
    niid: bool,
    balance: bool,
    partition: str,
    num_clusters: int,
):
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    train_path = os.path.join(output_dir, "train") + os.sep
    test_path = os.path.join(output_dir, "test") + os.sep

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    X, y, num_classes = _preprocess_tabular(csv_path, label_col)
    X_parts, y_parts, statistic = separate_data(
        (X, y),
        num_clients=num_clients,
        num_classes=num_classes,
        niid=niid,
        balance=balance,
        partition=partition,
        class_per_client=2,
    )
    train_data, test_data = split_data(X_parts, y_parts)
    extra_config = _build_uav_context(statistic, num_classes=num_classes, num_clusters=num_clusters)
    save_file(
        config_path,
        train_path,
        test_path,
        train_data,
        test_data,
        num_clients,
        num_classes,
        statistic,
        niid,
        balance,
        partition,
        extra_config=extra_config,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--label-col", type=str, default="label")
    ap.add_argument("--output-dir", type=str, default="TITS")
    ap.add_argument("--num-clients", type=int, default=50)
    ap.add_argument("--niid", type=_str2bool, default=True)
    ap.add_argument("--balance", type=_str2bool, default=False)
    ap.add_argument("--partition", type=str, default="dir", choices=["dir", "pat"])
    ap.add_argument("--num-clusters", type=int, default=3)
    args = ap.parse_args()

    generate_tits(
        csv_path=args.csv,
        output_dir=args.output_dir,
        label_col=args.label_col,
        num_clients=args.num_clients,
        niid=args.niid,
        balance=args.balance,
        partition=args.partition,
        num_clusters=args.num_clusters,
    )
