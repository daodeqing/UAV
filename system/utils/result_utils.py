import os

import h5py
import numpy as np


def average_data(algorithm="", dataset="", goal="", times=10, key="rs_test_acc"):
    metric_list = get_all_results_for_one_algo(algorithm, dataset, goal, times, key=key)

    max_metric = []
    for i in range(times):
        max_metric.append(metric_list[i].max())

    print(f"std for best {key}:", np.std(max_metric))
    print(f"mean for best {key}:", np.mean(max_metric))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10, key="rs_test_acc"):
    metric_list = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        metric_list.append(np.array(read_metric(file_name, key=key, delete=False)))

    return metric_list


def read_metric(file_name, key="rs_test_acc", delete=False):
    file_path = "../results/" + file_name + ".h5"

    with h5py.File(file_path, "r") as hf:
        arr = np.array(hf.get(key))

    if delete:
        os.remove(file_path)
    print("Length: ", len(arr))

    return arr


def read_metric_bundle(file_name, keys=None, delete=False):
    if keys is None:
        keys = [
            "rs_test_acc",
            "rs_test_macro_f1",
            "rs_test_weighted_f1",
            "rs_test_balanced_acc",
            "rs_test_macro_prauc",
            "rs_test_ece",
            "rs_test_brier",
            "rs_total_upload_bytes",
        ]

    file_path = "../results/" + file_name + ".h5"
    out = {}
    with h5py.File(file_path, "r") as hf:
        for key in keys:
            if key in hf:
                out[key] = np.array(hf.get(key))

    if delete:
        os.remove(file_path)
    return out


def read_data_then_delete(file_name, delete=False):
    return read_metric(file_name, key="rs_test_acc", delete=delete)
