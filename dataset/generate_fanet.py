import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler


LEVEL_MAP = {"LOW": 0.0, "MEDIUM": 0.5, "HIGH": 1.0}
TRAFFIC_MAP = {"CBR": 0.0, "Video": 1.0}
COMM_MAP = {"UAV <-> UAV": 0.0, "UAV <-> Base Station": 1.0, "Mixed": 0.5}


def _safe_split_count(n: int, ratio: float, min_keep: int):
    value = int(round(n * ratio))
    return int(max(1, min(value, max(n - min_keep, 1))))


def _save_npz(path, x, y):
    with open(path, "wb") as f:
        np.savez_compressed(f, data={"x": np.asarray(x, dtype=np.float32), "y": np.asarray(y, dtype=np.int64)})


def _label_stat(y):
    labels, counts = np.unique(y, return_counts=True)
    return [(int(label), int(count)) for label, count in zip(labels.tolist(), counts.tolist())]


def _prepare_attack_annotations(annotation_path: str):
    if annotation_path is None:
        return None
    ann = pd.read_csv(annotation_path)
    required = {"scenario_id", "node_id", "window_start", "label"}
    if not required.issubset(set(ann.columns)):
        raise ValueError(f"Attack annotation file must contain columns: {sorted(required)}")
    ann["scenario_id"] = ann["scenario_id"].astype(int)
    ann["node_id"] = ann["node_id"].astype(int)
    ann["window_start"] = ann["window_start"].astype(float)
    ann["label"] = ann["label"].astype(str)
    return ann


def _scenario_metadata(meta_row):
    comm_type = str(meta_row["communication_Type"]).strip()
    if "Base Station" in comm_type and "UAV" in comm_type and "<->" in comm_type and "Mixed" not in comm_type:
        comm_value = COMM_MAP["UAV <-> Base Station"]
    elif "UAV" in comm_type and "Base Station" not in comm_type:
        comm_value = COMM_MAP["UAV <-> UAV"]
    else:
        comm_value = COMM_MAP["Mixed"]
    return {
        "density_level": LEVEL_MAP.get(str(meta_row["node_density_level"]).strip().upper(), 0.5),
        "speed_level": LEVEL_MAP.get(str(meta_row["node_speed_level"]).strip().upper(), 0.5),
        "energy_level": LEVEL_MAP.get(str(meta_row["energy_capacity_level"]).strip().upper(), 0.5),
        "range_level": LEVEL_MAP.get(str(meta_row["range_level"]).strip().upper(), 0.5),
        "traffic_type": TRAFFIC_MAP.get(str(meta_row["traffic_type"]).strip(), 0.0),
        "communication_type": float(comm_value),
    }


def _prepare_scenario_dataframe(scenario_dir: Path, task_mode: str, annotations: pd.DataFrame = None):
    scenario_id = int(scenario_dir.name.split("_")[-1])
    meta = pd.read_csv(scenario_dir / "simulation_scenario.csv").iloc[0]
    meta_feat = _scenario_metadata(meta)

    node_qos = pd.read_csv(scenario_dir / "node_qos_metrics.csv")
    node_state = pd.read_csv(scenario_dir / "node_state.csv")
    packet_trace = pd.read_csv(scenario_dir / "packet_trace.csv")

    node_state["window_start"] = np.floor(node_state["time"].astype(float) - 1.0).clip(lower=0.0)
    node_state["energy_ratio"] = node_state["remaining_energy"].astype(float) / np.clip(
        node_state["initial_energy"].astype(float), 1e-6, None
    )
    state_agg = (
        node_state.groupby(["window_start", "node_id"], as_index=False)[
            ["energy_ratio", "remaining_energy", "speed_m_s", "pos_z", "vel_x", "vel_y", "vel_z"]
        ]
        .mean()
        .rename(columns={"node_id": "node_id"})
    )

    packet_trace["window_start"] = np.floor(packet_trace["TxTime_s"].astype(float))
    packet_trace["NodeIdTx"] = packet_trace["NodeIdTx"].astype(int)
    trace_agg = (
        packet_trace.groupby(["window_start", "NodeIdTx"], as_index=False)[
            ["PacketUid", "SNR_dB", "RSSI_dBm", "Link_Delay_ms"]
        ]
        .agg(
            tx_pkt_count=("PacketUid", "count"),
            snr_mean=("SNR_dB", "mean"),
            snr_std=("SNR_dB", "std"),
            rssi_mean=("RSSI_dBm", "mean"),
            delay_mean=("Link_Delay_ms", "mean"),
        )
        .rename(columns={"NodeIdTx": "node_id"})
    )

    df = node_qos.copy()
    df["window_start"] = df["window_start"].astype(float)
    df["node_id"] = df["node_id"].astype(int)
    df = df.merge(state_agg, on=["window_start", "node_id"], how="left")
    df = df.merge(trace_agg, on=["window_start", "node_id"], how="left")
    df = df.fillna(0.0)

    for key, value in meta_feat.items():
        df[key] = float(value)

    df["scenario_id"] = int(scenario_id)
    df["node_uid"] = df["scenario_id"].astype(str) + "_" + df["node_id"].astype(str)

    if task_mode == "scenario":
        df["target_label"] = int(scenario_id - 1)
        label_map = {str(i - 1): f"Scenario_{i}" for i in range(1, 9)}
    elif task_mode == "ids":
        if annotations is None:
            raise RuntimeError(
                "FANET IDS mode requires an attack annotation CSV with columns: scenario_id,node_id,window_start,label."
            )
        ann = annotations[annotations["scenario_id"] == scenario_id].copy()
        if ann.empty:
            raise RuntimeError(f"No annotations found for FANET scenario {scenario_id}.")
        df = df.merge(ann, on=["scenario_id", "node_id", "window_start"], how="left")
        df = df.dropna(subset=["label"]).copy()
        encoder = LabelEncoder()
        df["target_label"] = encoder.fit_transform(df["label"].astype(str))
        label_map = {str(idx): label for idx, label in enumerate(encoder.classes_.tolist())}
    else:
        raise ValueError(f"Unsupported task mode: {task_mode}")

    feature_cols = [
        "window_end",
        "sent_pkts",
        "sent_bytes",
        "recv_pkts",
        "recv_bytes",
        "avg_delay_ms",
        "jitter_ms",
        "throughput_bps",
        "DestinationRecvDataPkts",
        "DestinationRecvDataBytes",
        "goodput_bps",
        "energy_ratio",
        "remaining_energy",
        "speed_m_s",
        "pos_z",
        "vel_x",
        "vel_y",
        "vel_z",
        "tx_pkt_count",
        "snr_mean",
        "snr_std",
        "rssi_mean",
        "delay_mean",
        "density_level",
        "speed_level",
        "energy_level",
        "range_level",
        "traffic_type",
        "communication_type",
    ]
    df = df.sort_values(["node_id", "window_start"]).reset_index(drop=True)
    return df[["scenario_id", "node_id", "node_uid", "window_start", "target_label", *feature_cols]], label_map


def _pack_units_by_node(units, num_clients):
    bins = [[] for _ in range(num_clients)]
    loads = [0 for _ in range(num_clients)]
    for unit in sorted(units, key=lambda item: len(item[1]), reverse=True):
        idx = int(np.argmin(np.asarray(loads)))
        bins[idx].append(unit)
        loads[idx] += len(unit[1])
    return [bucket for bucket in bins if len(bucket) > 0]


def _client_profile(raw_df: pd.DataFrame):
    throughput = float(raw_df["throughput_bps"].mean())
    jitter = float(raw_df["jitter_ms"].mean())
    delay = float(raw_df["avg_delay_ms"].mean())
    energy = float(raw_df["energy_ratio"].mean())
    speed = float(raw_df["speed_m_s"].mean())
    snr = float(raw_df["snr_mean"].mean())
    recv_ratio = float(raw_df["recv_pkts"].sum() / max(raw_df["sent_pkts"].sum(), 1.0))
    return {
        "throughput": throughput,
        "jitter": jitter,
        "delay": delay,
        "energy_ratio": energy,
        "speed": speed,
        "snr": snr,
        "recv_ratio": recv_ratio,
    }


def _normalize_profiles(raw_profiles):
    keys = ["throughput", "jitter", "delay", "energy_ratio", "speed", "snr", "recv_ratio"]
    arr = np.asarray([[float(profile[k]) for k in keys] for profile in raw_profiles], dtype=np.float64)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    span = np.clip(maxs - mins, 1e-8, None)
    norm = (arr - mins) / span

    profiles = []
    for idx, raw in enumerate(raw_profiles):
        n_throughput, n_jitter, n_delay, n_energy, n_speed, n_snr, n_recv = norm[idx].tolist()
        compute_capacity = float(np.clip(0.45 * n_throughput + 0.25 * (1.0 - n_speed) + 0.30 * n_recv, 0.05, 0.99))
        link_quality = float(np.clip(0.45 * n_snr + 0.30 * (1.0 - n_delay) + 0.25 * (1.0 - n_jitter), 0.05, 0.99))
        energy_level = float(np.clip(0.70 * n_energy + 0.30 * (1.0 - n_throughput), 0.05, 0.99))
        expected_uptime = float(np.clip(0.35 * compute_capacity + 0.35 * link_quality + 0.30 * energy_level, 0.05, 0.99))
        profiles.append(
            {
                "compute_capacity": compute_capacity,
                "link_quality": link_quality,
                "energy_level": energy_level,
                "expected_uptime": expected_uptime,
                "availability": expected_uptime,
                "dropout_risk": float(np.clip(1.0 - expected_uptime, 0.01, 0.95)),
                "mobility_score": float(np.clip(1.0 - n_speed, 0.0, 1.0)),
                "burstiness": float(np.clip(raw["jitter"], 0.0, 1e6) / (1.0 + float(np.clip(raw["jitter"], 0.0, 1e6)))),
                "throughput": float(raw["throughput"]),
                "delay": float(raw["delay"]),
                "snr": float(raw["snr"]),
                "recv_ratio": float(raw["recv_ratio"]),
                "profile_source": "data_driven_from_fanet_node_state_and_qos",
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
    prof_keys = ["compute_capacity", "link_quality", "energy_level", "availability", "mobility_score"]
    prof = np.asarray([[float(profile[k]) for k in prof_keys] for profile in profiles], dtype=np.float32)
    feats = np.concatenate([hist, prof], axis=1)
    cluster_count = int(max(1, min(num_clusters, len(profiles))))
    if cluster_count <= 1:
        return [0 for _ in profiles]
    return KMeans(n_clusters=cluster_count, random_state=0, n_init=10).fit_predict(feats).tolist()


def generate_fanet(
    raw_dir: str,
    output_dir: str,
    num_clients: int,
    public_val_ratio: float,
    client_test_ratio: float,
    task_mode: str,
    annotation_path: str,
    num_clusters: int,
):
    raw_root = Path(raw_dir)
    output_root = Path(output_dir)
    (output_root / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "test").mkdir(parents=True, exist_ok=True)

    annotations = _prepare_attack_annotations(annotation_path)
    scenario_dirs = sorted([path for path in raw_root.iterdir() if path.is_dir() and path.name.startswith("Scenario_")])

    unit_frames = []
    attack_label_map = None
    for scenario_dir in scenario_dirs:
        df, label_map = _prepare_scenario_dataframe(scenario_dir, task_mode=task_mode, annotations=annotations)
        attack_label_map = label_map
        for unit_key, unit_df in df.groupby("node_uid", sort=False):
            unit_frames.append((unit_key, unit_df.reset_index(drop=True)))

    if len(unit_frames) == 0:
        raise RuntimeError("No FANET units were generated. Check task mode and annotation inputs.")

    public_parts = []
    processed_units = []
    for unit_key, unit_df in unit_frames:
        if len(unit_df) < 6:
            continue
        n_public = _safe_split_count(len(unit_df), public_val_ratio, min_keep=4)
        public_parts.append(unit_df.iloc[:n_public].copy())
        processed_units.append((unit_key, unit_df.iloc[n_public:].copy()))

    packed_clients = _pack_units_by_node(processed_units, num_clients=num_clients)
    raw_profiles = []
    train_data = []
    test_data = []
    statistic = []
    client_indices = {}

    for cid, client_units in enumerate(packed_clients):
        train_frames = []
        test_frames = []
        meta_units = []
        profile_frames = []
        for unit_key, unit_df in client_units:
            if len(unit_df) < 4:
                continue
            n_test = _safe_split_count(len(unit_df), client_test_ratio, min_keep=3)
            train_frames.append(unit_df.iloc[:-n_test].copy())
            test_frames.append(unit_df.iloc[-n_test:].copy())
            profile_frames.append(unit_df.copy())
            meta_units.append(
                {
                    "unit_key": unit_key,
                    "scenario_id": int(unit_df["scenario_id"].iloc[0]),
                    "node_id": int(unit_df["node_id"].iloc[0]),
                    "rows": int(len(unit_df)),
                }
            )
        if len(train_frames) == 0 or len(test_frames) == 0:
            continue

        train_df = pd.concat(train_frames, axis=0, ignore_index=True)
        test_df = pd.concat(test_frames, axis=0, ignore_index=True)
        feature_cols = [col for col in train_df.columns if col not in {"scenario_id", "node_id", "node_uid", "window_start", "target_label"}]
        train_data.append({"x": train_df[feature_cols].to_numpy(dtype=np.float32), "y": train_df["target_label"].to_numpy(dtype=np.int64)})
        test_data.append({"x": test_df[feature_cols].to_numpy(dtype=np.float32), "y": test_df["target_label"].to_numpy(dtype=np.int64)})
        statistic.append(_label_stat(train_df["target_label"].to_numpy(dtype=np.int64)))
        raw_profiles.append(_client_profile(pd.concat(profile_frames, axis=0, ignore_index=True)))
        client_indices[str(len(train_data) - 1)] = {"units": meta_units}

    if len(train_data) == 0:
        raise RuntimeError("No FANET clients were generated after filtering short node traces.")

    scaler = StandardScaler()
    scaler.fit(np.concatenate([item["x"] for item in train_data], axis=0))
    for item in train_data:
        item["x"] = scaler.transform(item["x"]).astype(np.float32)
    for item in test_data:
        item["x"] = scaler.transform(item["x"]).astype(np.float32)

    public_df = pd.concat(public_parts, axis=0, ignore_index=True)
    feature_cols = [col for col in public_df.columns if col not in {"scenario_id", "node_id", "node_uid", "window_start", "target_label"}]
    public_x = scaler.transform(public_df[feature_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    public_y = public_df["target_label"].to_numpy(dtype=np.int64)

    normalized_profiles = _normalize_profiles(raw_profiles)
    client_clusters = _cluster_clients(statistic, normalized_profiles, num_clusters=num_clusters)
    for cid, cluster_id in enumerate(client_clusters):
        normalized_profiles[cid]["cluster_id"] = int(cluster_id)

    feature_dim = int(train_data[0]["x"].shape[1])
    num_classes = int(max(int(k) for k in attack_label_map.keys()) + 1)
    config = {
        "num_clients": int(len(train_data)),
        "num_classes": num_classes,
        "feature_dim": feature_dim,
        "non_iid": True,
        "balance": False,
        "partition": "by_node",
        "client_partition": "by_node",
        "Size of samples for labels in clients": statistic,
        "client_profiles": normalized_profiles,
        "client_clusters": [int(c) for c in client_clusters],
        "public_val_indices": [{"rows": int(len(public_y)), "source": "per-node prefix windows"}],
        "attack_label_map": attack_label_map,
        "profile_source": "data_driven_from_fanet_node_state_and_qos",
        "task_mode": task_mode,
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

    print(f"Saved FANET dataset to {output_root}")
    print(f"Clients: {len(train_data)}, public_val: {len(public_y)}, feature_dim: {feature_dim}, classes: {num_classes}")
    if task_mode == "scenario":
        print("Note: local FANET_Dataset_NS3.40 does not contain IDS attack labels; scenario-mode was used.")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=str, default=str(base_dir / "FANET_Dataset_NS3.40"))
    ap.add_argument("--output-dir", type=str, default=str(base_dir / "FANET"))
    ap.add_argument("--num-clients", type=int, default=20)
    ap.add_argument("--public-val-ratio", type=float, default=0.05)
    ap.add_argument("--client-test-ratio", type=float, default=0.2)
    ap.add_argument("--task-mode", type=str, default="scenario", choices=["scenario", "ids"])
    ap.add_argument("--attack-annotation", type=str, default=None)
    ap.add_argument("--num-clusters", type=int, default=3)
    args = ap.parse_args()

    generate_fanet(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        num_clients=args.num_clients,
        public_val_ratio=args.public_val_ratio,
        client_test_ratio=args.client_test_ratio,
        task_mode=args.task_mode,
        annotation_path=args.attack_annotation,
        num_clusters=args.num_clusters,
    )
