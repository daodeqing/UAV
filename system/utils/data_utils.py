import json
import os

import numpy as np
import torch


_DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset"))


def dataset_dir(dataset):
    return os.path.join(_DATASET_ROOT, dataset)


def dataset_config_path(dataset):
    return os.path.join(dataset_dir(dataset), "config.json")


def load_dataset_config(dataset):
    cfg_path = dataset_config_path(dataset)
    if not os.path.exists(cfg_path):
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _split_file(dataset, split, idx=None):
    base_dir = os.path.join(dataset_dir(dataset), split)
    if idx is None:
        return os.path.join(dataset_dir(dataset), f"{split}.npz")
    return os.path.join(base_dir, f"{idx}.npz")


def read_data(dataset, idx, is_train=True):
    if is_train:
        train_file = _split_file(dataset, "train", idx)
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_file = _split_file(dataset, "test", idx)
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, idx, is_train=True):

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data

def read_client_data_un(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train_float = train_data['x'].astype(float)
        X_train = torch.tensor(X_train_float, dtype=torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test_float = test_data['x'].astype(float)
        X_test = torch.tensor(X_test_float, dtype=torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def read_public_data(dataset, split="public_val"):
    split_file = _split_file(dataset, split, idx=None)
    if not os.path.exists(split_file):
        return []

    with open(split_file, "rb") as f:
        data = np.load(f, allow_pickle=True)["data"].tolist()

    X_data = np.asarray(data.get("x", []), dtype=np.float32)
    y_data = np.asarray(data.get("y", []), dtype=np.int64)
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.int64)
    return [(x, y) for x, y in zip(X_tensor, y_tensor)]
