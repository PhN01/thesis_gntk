import os
import argparse
import time
import pickle

import sys

sys.path.append(".")

from src.gntk import gntk
from src.data import data_loaders
from src.utils import utils
from src.config import config as cfg
from src.utils import logging
from src.data import graph_utils

"""
Routine to calculate a single gram matrix for the dataset and parameters
parsed through the command line.

Example call:
bsub -n 20 -W 4:00 -R "rusage[mem=8096]" python run_gram_single_v2.py
--data_dir ../../data/TU_DO --dataset PROTEINS --exp_name standard_params_new
--L 1 --R 1 --scale uniform --jk 1 --n_jobs 40 --scratch 1 --timing 0
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ds",
        "--data_source",
        default="TU_DO",
        help="Source of the data. In ['TU_DO','Du_et_al']",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="MUTAG",
        help="Dataset for which to calculate the gram matrix",
    )
    parser.add_argument(
        "-L",
        "--n_blocks",
        default=1,
        type=int,
        help="GNTK parameter: number of block layers",
    )
    parser.add_argument(
        "-R",
        "--n_fc_layers",
        default=1,
        type=int,
        help="GNTK parameter: number of fully connected layers in the MLP layer",
    )
    parser.add_argument(
        "-s",
        "--scale",
        default="degree",
        help="GNTK parameter: normalization type for aggregation step",
    )
    parser.add_argument(
        "-jk",
        "--jumping_knowledge",
        default=1,
        type=int,
        help="GNTK parameter: jumping knowledge",
    )
    args = parser.parse_args()

    config = cfg.Config()
    dataset_out_dir = f"{config.matrix_path}/GNTK_{args.data_source}/{args.dataset}"

    gram_name = f"gram_{args.dataset}_L{args.n_blocks}_R{args.n_fc_layers}_scale{args.scale}_jk{args.jumping_knowledge}"

    # prepare filesystem
    # specify output directory
    out_dir = f"{dataset_out_dir}/{gram_name}"
    # create output directory if non-existent
    utils.make_dirs_checked(out_dir)

    logger = logging.get_logger(out_dir)

    assert not os.path.isfile(
        f"{out_dir}/gram.pkl"
    ), "Gram matrix exists already. Stopping process."

    # load and prepare graph data
    logger.info("Loading data")
    graphs, labels = data_loaders.load_graphs(
        source=args.data_source, dataset=args.dataset
    )

    n_graphs = len(graphs)

    # create list of one-hot encoded label matrices and list of adjacency matrices
    lab_list = graph_utils.node_lab_mat(graphs)
    adj_list = [graph_utils.get_graph_adj(graph) for graph in graphs]

    logger.info(f"Computing gram matrix for dataset {args.dataset}")
    logger.info(f"{gram_name}")

    # start cpu timer
    start_time = time.process_time()

    # run gram matrix calculation
    gntk_instance = gntk.GNTK(
        L=args.n_blocks, R=args.n_fc_layers, scale=args.scale, jk=args.jumping_knowledge
    )
    gram_mat = gntk_instance.gntk_gram(lab_list, adj_list)

    # stop cpu timer
    stop_time = time.process_time()

    logger.info(f"Job finished. Elapsed time: {stop_time-start_time}")

    with open(f"{out_dir}/gram.pkl", "wb") as f:
        pickle.dump(gram_mat, f)
