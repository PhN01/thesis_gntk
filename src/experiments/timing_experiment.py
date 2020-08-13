import json
import graphkernels as gk
import numpy as np
import argparse
from tqdm import tqdm
import os

import sys

sys.path.append(".")

from src.utils import utils
from src.gntk import gntk
from src.config import config as cfg
from src.data import data_loaders
from src.data import graph_utils

from timeit import default_timer as timer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--n_samples",
        default=50,
        type=int,
        help="Number of samples to calculate the gram matrices for.",
    )

    args = parser.parse_args()
    config = cfg.Config()
    exp_config = cfg.TimingExpConfig()

    out_dir = f"{config.exp_path}/timing"
    utils.make_dirs_checked(out_dir)

    datasets = ["IMDBBINARY", "IMDBMULTI", "MUTAG", "NCI1", "PROTEINS", "PTC"]
    kernels = ["GNTK", "VH", "EH", "SP", "WL"]

    gram_time = {dataset: {} for dataset in datasets}
    for dataset in tqdm(datasets, desc="Datasets"):

        # load data
        graphs, labels = data_loaders.load_graphs_tudortmund(dataset)

        n_graphs = len(graphs)

        np.random.seed(42)
        sample_indices = np.random.choice(
            range(n_graphs), args.n_samples, replace=False
        )

        graphs = [graphs[i] for i in sample_indices]
        igraphs = graph_utils.conv_graph_nx2ig(graphs)

        for kernel in tqdm(kernels, desc="Kernels"):

            if kernel == "GNTK":

                lab_list = graph_utils.node_lab_mat(graphs)
                adj_list = [graph_utils.get_graph_adj(graph) for graph in graphs]

                params = exp_config.GNTK[dataset]

                start_time = timer()
                gntk_instance = gntk.GNTK(
                    L=params["n_blocks"],
                    R=params["n_fc_layers"],
                    scale=params["scale"],
                    jk=params["jk"],
                )
                mat = gntk_instance.gntk_gram(lab_list, adj_list, n_jobs=1, verbose=0)

                gram_time[dataset][kernel] = timer() - start_time

            elif kernel == "VH":

                start_time = timer()
                mat = gk.CalculateEdgeHistKernel(igraphs)

                gram_time[dataset][kernel] = timer() - start_time

            elif kernel == "EH":

                start_time = timer()
                mat = gk.CalculateVertexHistKernel(igraphs)

                gram_time[dataset][kernel] = timer() - start_time

            elif kernel == "SP":

                start_time = timer()
                mat = gk.CalculateShortestPathKernel(igraphs)

                gram_time[dataset][kernel] = timer() - start_time

            elif kernel == "WL":
                try:
                    iterations = exp_config.WL[dataset]

                    start_time = timer()
                    mat = gk.CalculateWLKernel(igraphs, par=iterations)

                    gram_time[dataset][kernel] = timer() - start_time
                except:
                    gram_time[dataset][kernel] = 99999999

    gram_time_normalized = {}
    for dataset, kernel_times in gram_time.items():
        gram_time_normalized[dataset] = {}
        min_time = np.min(list(kernel_times.values()))
        for k, v in kernel_times.items():
            gram_time_normalized[dataset][k] = v / min_time

    with open(f"{out_dir}/gram_timing_{args.n_samples}samples.txt", "w") as f:
        json.dump(gram_time, f, indent=4)

    with open(
        f"{out_dir}/gram_timing_normalized_{args.n_samples}samples.txt", "w"
    ) as f:
        json.dump(gram_time_normalized, f, indent=4)
