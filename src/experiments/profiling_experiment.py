import os
import json

import numpy as np

import sys
import argparse

sys.path.append(".")

from tqdm import tqdm
import random

import line_profiler as lp

from src.gntk import gntk_profiling
from src.utils import utils
from src.data import data_loaders
from src.config import config as cfg
from src.data import graph_utils

import re


def analyse_profiler_output(output):
    """
    :param output: text output of the line profiler
    :return: dictionary with the time spend on particular process steps in percent of the total time
    """

    keys = [
        "Other",
        "Aggregation Normalization Matrix Calculation",
        "Kronecker Product Calculation",
        "Sigma Matrix Initialization",
        "Aggregate",
        "Update Sigma / Theta",
        "Readout",
    ]

    line_map = {
        "41": "Other",
        "42": "Other",
        "43": "Other",
        "44": "Other",
        "46": "Other",
        "49": "Other",
        "52": "Aggregation Normalization Matrix Calculation",
        "53": "Aggregation Normalization Matrix Calculation",
        "54": "Aggregation Normalization Matrix Calculation",
        "55": "Aggregation Normalization Matrix Calculation",
        "56": "Aggregation Normalization Matrix Calculation",
        "57": "Aggregation Normalization Matrix Calculation",
        "58": "Aggregation Normalization Matrix Calculation",
        "59": "Aggregation Normalization Matrix Calculation",
        "60": "Aggregation Normalization Matrix Calculation",
        "61": "Aggregation Normalization Matrix Calculation",
        "62": "Aggregation Normalization Matrix Calculation",
        "65": "Kronecker Product Calculation",
        "66": "Kronecker Product Calculation",
        "67": "Kronecker Product Calculation",
        "70": "Sigma Matrix Initilization",
        "71": "Sigma Matrix Initilization",
        "72": "Sigma Matrix Initilization",
        "73": "Sigma Matrix Initilization",
        "76": "Other",
        "77": "Other",
        "80": "Other",
        "83": "AGGREGATE",
        "84": "AGGREGATE",
        "86": "AGGREGATE",
        "89": "Update Sigma / Theta",
        "91": "Update Sigma / Theta",
        "92": "Update Sigma / Theta",
        "93": "Update Sigma / Theta",
        "95": "Update Sigma / Theta",
        "96": "Update Sigma / Theta",
        "99": "Update Sigma / Theta",
        "100": "Update Sigma / Theta",
        "101": "Update Sigma / Theta",
        "102": "Update Sigma / Theta",
        "105": "Update Sigma / Theta",
        "108": "Update Sigma / Theta",
        "109": "Update Sigma / Theta",
        "110": "Update Sigma / Theta",
        "112": "Update Sigma / Theta",
        "113": "Update Sigma / Theta",
        "116": "Update Sigma / Theta",
        "117": "Update Sigma / Theta",
        "120": "Update Sigma / Theta",
        "121": "Update Sigma / Theta",
        "122": "Update Sigma / Theta",
        "124": "Update Sigma / Theta",
        "125": "Update Sigma / Theta",
        "128": "Update Sigma / Theta",
        "129": "Update Sigma / Theta",
        "132": "Update Sigma / Theta",
        "135": "READOUT",
        "138": "Other",
        "139": "Other",
        "140": "Other",
        "142": "Other",
    }

    res = {key: 0 for key in keys}

    for line in output.split("\n"):
        clean = re.sub("\s+", r",", line.strip()).split(",")

        if clean[0] in line_map.keys():
            try:
                res[line_map[clean[0]]] = res[line_map[clean[0]]] + float(clean[2])
            except:
                pass

    total = np.sum(list(res.values()))

    res = {key: value / total * 100 for key, value in res.items()}

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--n_samples",
        default=10,
        type=int,
        help="Number of samples to draw from each dataset",
    )
    args = parser.parse_args()

    config = cfg.Config()

    L_list = [1, 3, 10]
    n = args.n_samples
    R = 3

    out_dir = f"{config.exp_path}/GNTK/profiling"
    utils.make_dirs_checked(out_dir)
    profiles_dir = f"{out_dir}/{n}_samples"
    utils.make_dirs_checked(profiles_dir)

    datasets = ["MUTAG", "PTC", "IMDBBINARY", "IMDBMULTI", "PROTEINS", "NCI1", "COLLAB"]

    data = {}
    for dataset in tqdm(datasets, desc="Loading datasets"):
        data[dataset] = {}

        graphs, labels = data_loaders.load_graphs_tudortmund(dataset=dataset)
        data[dataset]["graphs"] = graphs
        data[dataset]["labels"] = labels

    results = {}
    for L in L_list:
        results[f"n{n}_L{L}_R{R}"] = {}
        for dataset in tqdm(datasets, desc=f"Profiler // n{n}_L{L}_R{R}"):

            graphs = data[dataset]["graphs"]
            labels = data[dataset]["labels"]
            n_graphs = len(graphs)

            random.seed(42)
            samples = random.sample(range(n_graphs), n)

            graphs = [graphs[i] for i in samples]

            lab_list = graph_utils.node_lab_mat(graphs)
            adj_list = [graph_utils.get_graph_adj(graph) for graph in graphs]

            profiler = lp.LineProfiler(gntk_profiling.gntk_gram_profiling)

            with open(f"{profiles_dir}/{dataset}_n{n}_L{L}_R{R}_profile.txt", "w") as f:
                profiler.run(
                    f'gntk_profiling.gntk_gram_profiling(lab_list, adj_list, {L}, {R}, "degree", 1)'
                ).print_stats(f)

            with open(f"{profiles_dir}/{dataset}_n{n}_L{L}_R{R}_profile.txt", "r") as f:
                out = f.read()

            results[f"n{n}_L{L}_R{R}"][dataset] = analyse_profiler_output(out)

    with open(f"{out_dir}/results_{n}samples.txt", "w") as f:
        json.dump(results, f, indent=4)
