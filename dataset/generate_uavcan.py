import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


SCENARIO_ATTACK_FAMILY = {
    1: "Flooding",
    2: "Flooding",
    3: "Fuzzy",
    4: "Fuzzy",
    5: "Replay",
    6: "Replay",
    7: "MixedFloodingFuzzy",
    8: "MixedFuzzyReplay",
    9: "MixedFloodingReplay",
    10: "MixedAll",
}

LINE_RE = re.compile(
    r"^(?P<label>Normal|Attack)\s+\((?P<ts>\d+\.\d+)\)\s+\S+\s+(?P<can_id>[0-9A-F]+)\s+\[(?P<dlc>\d+)\]\s*(?P<data>.*)$"
)


@dataclass
class SegmentBundle:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    public_meta: dict
    client_meta: dict
    raw_profile: dict


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _parse_uavcan_file(file_path: Path, scenario_id: int, task_mode: str):
    rows = []
    last_ts = None
    last_ts_by_id = {}

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = LINE_RE.match(line.strip())
            if match is None:
                continue

            raw_label = match.group("label")
            if task_mode == "attack_family" and scenario_id > 6 and raw_label == "Attack":
                continue

            ts = float(match.group("ts"))
            can_hex = match.group("can_id")
            can_id = int(can_hex, 16)
            dlc = int(match.group("dlc"))
            payload_tokens = [tok for tok in match.group("data").strip().split() if tok]
            payload = [int(tok, 16) for tok in payload_tokens[:8]]
            while len(payload) < 8:
                payload.append(0)

            delta_t = 0.0 if last_ts is None else max(ts - last_ts, 0.0)
            last_ts = ts
            can_cycle = 0.0 if can_id not in last_ts_by_id else max(ts - last_ts_by_id[can_id], 0.0)
            last_ts_by_id[can_id] = ts

            if task_mode == "binary":
                label = 0 if raw_label == "Normal" else 1
            elif task_mode == "attack_family":
                if raw_label == "Normal":
                    label = 0
                else:
                    family = SCENARIO_ATTACK_FAMILY[scenario_id]
                    label = {"Flooding": 1, "Fuzzy": 2, "Replay": 3}[family]
            else:
                raise ValueError(f"Unsupported task mode: {task_mode}")

            payload_arr = np.asarray(payload, dtype=np.float32)
            rows.append(
                [
                    ts,
                    delta_t,
                    float(scenario_id),
                    float(can_id),
                    float(int(can_hex[:2], 16)),
                    float(dlc),
                    float(payload_arr.sum()),
                    float(payload_arr.mean()),
                    float(payload_arr.std()),
                    float(payload_arr.max()),
                    float(payload_arr.min()),
                    float(can_cycle),
                    float(raw_label == "Attack"),
                    *payload_arr.tolist(),
                    float(label),
                ]
            )

    if len(rows) == 0:
        return np.zeros((0, 20), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    arr = np.asarray(rows, dtype=np.float32)
    X = arr[:, :-1]
    y = arr[:, -1].astype(np.int64)
    return X, y


def _allocate_quotas(num_clients: int, scenario_ids):
    base = num_clients // max(len(scenario_ids), 1)
    rem = num_clients % max(len(scenario_ids), 1)
    quotas = {}
    for idx, sid in enumerate(sorted(scenario_ids)):
        quotas[int(sid)] = int(base + (1 if idx < rem else 0))
    return quotas


def _safe_split_count(n: int, ratio: float, min_keep: int):
    value = int(round(n * ratio))
    return int(max(1, min(value, max(n - min_keep, 1))))


def _segment_profile(segment_x: np.ndarray, segment_y: np.ndarray, scenario_id: int):
    delta_t = segment_x[:, 1]
    dlc = segment_x[:, 5]
    payload_std = segment_x[:, 8]
    can_cycle = segment_x[:, 11]
    attack_flag = segment_x[:, 12]
    can_ids = segment_x[:, 3]

    duration = float(max(segment_x[-1, 0] - segment_x[0, 0], 1e-6))
    msg_rate = float(len(segment_x) / duration)
    mean_dlc = float(np.mean(dlc))
    cycle_mean = float(np.mean(can_cycle))
    cycle_jitter = float(np.std(can_cycle))
    burstiness = float(np.std(delta_t) / max(np.mean(delta_t), 1e-6))
    payload_dispersion = float(np.mean(payload_std))
    unique_can_ratio = float(len(np.unique(can_ids)) / max(len(segment_x), 1))
    attack_exposure = float(np.mean(attack_flag))

    return {
        "scenario_id": int(scenario_id),
        "msg_rate": msg_rate,
        "mean_dlc": mean_dlc,
        "cycle_mean": cycle_mean,
        "cycle_jitter": cycle_jitter,
        "burstiness": burstiness,
        "payload_dispersion": payload_dispersion,
        "unique_can_ratio": unique_can_ratio,
        "attack_exposure": attack_exposure,
        "label_entropy": float(
            -np.sum(
                (np.bincount(segment_y, minlength=int(segment_y.max()) + 1) / max(len(segment_y), 1))
                * np.log(
                    np.clip(
                        np.bincount(segment_y, minlength=int(segment_y.max()) + 1) / max(len(segment_y), 1),
                        1e-12,
                        1.0,
                    )
                )
            )
        ),
    }


def _normalize_profiles(raw_profiles):
    if len(raw_profiles) == 0:
        return []

    keys = [
        "msg_rate",
        "mean_dlc",
        "cycle_mean",
        "cycle_jitter",
        "burstiness",
        "payload_dispersion",
        "unique_can_ratio",
        "attack_exposure",
    ]
    arr = np.asarray([[float(profile[k]) for k in keys] for profile in raw_profiles], dtype=np.float64)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    span = np.clip(maxs - mins, 1e-8, None)
    norm = (arr - mins) / span

    profiles = []
    for idx, raw in enumerate(raw_profiles):
        n_msg, n_dlc, n_cycle_mean, n_cycle_jitter, n_burst, n_payload, n_unique, n_attack = norm[idx].tolist()
        compute_capacity = float(np.clip(0.45 * (1.0 - n_burst) + 0.35 * (1.0 - n_attack) + 0.20 * n_unique, 0.05, 0.99))
        link_quality = float(np.clip(0.55 * (1.0 - n_cycle_jitter) + 0.25 * (1.0 - n_cycle_mean) + 0.20 * (1.0 - n_payload), 0.05, 0.99))
        energy_level = float(np.clip(0.60 * (1.0 - n_msg) + 0.25 * (1.0 - n_dlc) + 0.15 * (1.0 - n_attack), 0.05, 0.99))
        expected_uptime = float(np.clip(0.35 * compute_capacity + 0.35 * link_quality + 0.30 * energy_level, 0.05, 0.99))
        profiles.append(
            {
                "scenario_id": int(raw["scenario_id"]),
                "compute_capacity": compute_capacity,
                "link_quality": link_quality,
                "energy_level": energy_level,
                "expected_uptime": expected_uptime,
                "availability": expected_uptime,
                "dropout_risk": float(np.clip(1.0 - expected_uptime, 0.01, 0.95)),
                "mobility_score": float(np.clip(1.0 - n_cycle_mean, 0.0, 1.0)),
                "burstiness": float(np.clip(raw["burstiness"], 0.0, 1e6) / (1.0 + float(np.clip(raw["burstiness"], 0.0, 1e6)))),
                "attack_exposure": float(raw["attack_exposure"]),
                "bus_load": float(raw["msg_rate"]),
                "message_jitter": float(raw["cycle_jitter"]),
                "payload_dispersion": float(raw["payload_dispersion"]),
                "unique_can_ratio": float(raw["unique_can_ratio"]),
                "profile_source": "data_driven_from_uavcan_bus_statistics",
            }
        )
    return profiles


def _cluster_clients(statistic, profiles, num_clusters):
    num_classes = 1 + max((int(label) for client_stat in statistic for label, _ in client_stat), default=0)
    hist = np.zeros((len(statistic), num_classes), dtype=np.float32)
    for cid, client_stat in enumerate(statistic):
        for label, count in client_stat:
            hist[cid, int(label)] = float(count)
    totals = np.clip(hist.sum(axis=1, keepdims=True), 1.0, None)
    hist = hist / totals

    prof_keys = ["compute_capacity", "link_quality", "energy_level", "availability", "attack_exposure"]
    prof = np.asarray([[float(profile[k]) for k in prof_keys] for profile in profiles], dtype=np.float32)
    feats = np.concatenate([hist, prof], axis=1)
    cluster_count = int(max(1, min(num_clusters, len(profiles))))
    if cluster_count <= 1:
        return [0 for _ in profiles]
    return KMeans(n_clusters=cluster_count, random_state=0, n_init=10).fit_predict(feats).tolist()


def _label_stat(y):
    labels, counts = np.unique(y, return_counts=True)
    return [(int(label), int(count)) for label, count in zip(labels.tolist(), counts.tolist())]


def _save_npz(path, x, y):
    with open(path, "wb") as f:
        np.savez_compressed(f, data={"x": np.asarray(x, dtype=np.float32), "y": np.asarray(y, dtype=np.int64)})


def generate_uavcan(
    raw_dir: str,
    output_dir: str,
    num_clients: int,
    public_val_ratio: float,
    client_test_ratio: float,
    task_mode: str,
    num_clusters: int,
):
    raw_root = Path(raw_dir)
    output_root = Path(output_dir)
    (output_root / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "test").mkdir(parents=True, exist_ok=True)

    if task_mode == "binary":
        scenario_ids = list(range(1, 11))
        attack_label_map = {0: "Normal", 1: "Attack"}
    elif task_mode == "attack_family":
        scenario_ids = list(range(1, 7))
        attack_label_map = {0: "Normal", 1: "Flooding", 2: "Fuzzy", 3: "Replay"}
    else:
        raise ValueError(f"Unsupported task mode: {task_mode}")

    quotas = _allocate_quotas(num_clients, scenario_ids)
    client_bundles = []
    public_val_parts = []
    public_val_meta = []

    for scenario_id in scenario_ids:
        file_path = raw_root / f"type{scenario_id}_label.bin"
        X, y = _parse_uavcan_file(file_path, scenario_id=scenario_id, task_mode=task_mode)
        if len(y) < 10:
            continue

        n_public = _safe_split_count(len(y), public_val_ratio, min_keep=max(8, quotas[scenario_id] * 4))
        public_x = X[:n_public]
        public_y = y[:n_public]
        public_val_parts.append((public_x, public_y))
        public_val_meta.append({"scenario_id": int(scenario_id), "count": int(len(public_y))})

        remain_x = X[n_public:]
        remain_y = y[n_public:]
        quota = max(1, quotas[scenario_id])
        seg_edges = np.linspace(0, len(remain_y), quota + 1, dtype=int)
        for seg_idx in range(quota):
            start = int(seg_edges[seg_idx])
            end = int(seg_edges[seg_idx + 1])
            seg_x = remain_x[start:end]
            seg_y = remain_y[start:end]
            if len(seg_y) < 8:
                continue

            n_test = _safe_split_count(len(seg_y), client_test_ratio, min_keep=4)
            train_x = seg_x[:-n_test]
            train_y = seg_y[:-n_test]
            test_x = seg_x[-n_test:]
            test_y = seg_y[-n_test:]
            raw_profile = _segment_profile(seg_x, seg_y, scenario_id=scenario_id)
            client_bundles.append(
                SegmentBundle(
                    train_x=train_x,
                    train_y=train_y,
                    test_x=test_x,
                    test_y=test_y,
                    public_meta={"scenario_id": int(scenario_id), "public_count": int(len(public_y))},
                    client_meta={
                        "scenario_id": int(scenario_id),
                        "source_file": file_path.name,
                        "segment_index": int(seg_idx),
                        "raw_start": int(start + n_public),
                        "raw_end": int(end + n_public),
                    },
                    raw_profile=raw_profile,
                )
            )

    if len(client_bundles) == 0:
        raise RuntimeError("No UAVCAN client segments were generated. Check task mode and raw dataset path.")

    client_bundles = client_bundles[:num_clients]
    raw_profiles = [bundle.raw_profile for bundle in client_bundles]
    train_all = np.concatenate([bundle.train_x for bundle in client_bundles], axis=0)
    scaler = StandardScaler()
    scaler.fit(train_all)

    train_data = []
    test_data = []
    statistic = []
    client_indices = {}
    normalized_profiles = _normalize_profiles(raw_profiles)

    for cid, bundle in enumerate(client_bundles):
        train_x = scaler.transform(bundle.train_x).astype(np.float32)
        test_x = scaler.transform(bundle.test_x).astype(np.float32)
        train_y = bundle.train_y.astype(np.int64)
        test_y = bundle.test_y.astype(np.int64)
        train_data.append({"x": train_x, "y": train_y})
        test_data.append({"x": test_x, "y": test_y})
        statistic.append(_label_stat(train_y))
        client_indices[str(cid)] = {
            **bundle.client_meta,
            "train_samples": int(len(train_y)),
            "test_samples": int(len(test_y)),
        }

    public_x = np.concatenate([part[0] for part in public_val_parts], axis=0)
    public_y = np.concatenate([part[1] for part in public_val_parts], axis=0)
    public_x = scaler.transform(public_x).astype(np.float32)
    public_y = public_y.astype(np.int64)

    client_clusters = _cluster_clients(statistic, normalized_profiles, num_clusters=num_clusters)
    for cid, cluster_id in enumerate(client_clusters):
        normalized_profiles[cid]["cluster_id"] = int(cluster_id)

    feature_dim = int(train_data[0]["x"].shape[1])
    num_classes = int(max(attack_label_map.keys()) + 1)
    config = {
        "num_clients": int(len(train_data)),
        "num_classes": num_classes,
        "feature_dim": feature_dim,
        "non_iid": True,
        "balance": False,
        "partition": "by_session",
        "client_partition": "by_session",
        "Size of samples for labels in clients": statistic,
        "client_profiles": normalized_profiles,
        "client_clusters": [int(c) for c in client_clusters],
        "public_val_indices": public_val_meta,
        "attack_label_map": {str(k): v for k, v in attack_label_map.items()},
        "profile_source": "data_driven_from_uavcan_bus_statistics",
        "task_mode": task_mode,
        "scenario_attack_family": {str(k): v for k, v in SCENARIO_ATTACK_FAMILY.items()},
        "batch_size": 10,
        "alpha": 0.0,
    }

    for cid, train_dict in enumerate(train_data):
        _save_npz(output_root / "train" / f"{cid}.npz", train_dict["x"], train_dict["y"])
    for cid, test_dict in enumerate(test_data):
        _save_npz(output_root / "test" / f"{cid}.npz", test_dict["x"], test_dict["y"])
    _save_npz(output_root / "public_val.npz", public_x, public_y)
    _save_npz(output_root / "val.npz", public_x, public_y)

    with (output_root / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False)
    with (output_root / "client_indices.json").open("w", encoding="utf-8") as f:
        json.dump(client_indices, f, ensure_ascii=False)

    print(f"Saved UAVCAN dataset to {output_root}")
    print(f"Clients: {len(train_data)}, public_val: {len(public_y)}, feature_dim: {feature_dim}, classes: {num_classes}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw-dir",
        type=str,
        default=str(base_dir / "UAVCAN_Attack_dataset(DroneCAN)"),
    )
    ap.add_argument("--output-dir", type=str, default=str(base_dir / "UAVCAN"))
    ap.add_argument("--num-clients", type=int, default=20)
    ap.add_argument("--public-val-ratio", type=float, default=0.05)
    ap.add_argument("--client-test-ratio", type=float, default=0.2)
    ap.add_argument("--task-mode", type=str, default="binary", choices=["binary", "attack_family"])
    ap.add_argument("--num-clusters", type=int, default=3)
    args = ap.parse_args()

    generate_uavcan(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        num_clients=args.num_clients,
        public_val_ratio=args.public_val_ratio,
        client_test_ratio=args.client_test_ratio,
        task_mode=args.task_mode,
        num_clusters=args.num_clusters,
    )
