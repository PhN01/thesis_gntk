import os
import numpy as np
import logging as _logging
from tqdm import tqdm
import scipy.sparse as sparse
import pickle

import networkx as nx
import igraph as ig

import sys

sys.path.append(".")

from src.config import config as cfg
from src.utils import logging

logger = _logging.getLogger(__name__)


def load_graphs(source="TU_DO", dataset="MUTAG"):
    graph_loaders = {"TU_DO": load_graphs_tudortmund, "Du_et_al": load_graphs_duetal}
    g_list, g_labels = graph_loaders[source](dataset)

    return (g_list, g_labels)


def load_graphs_tudortmund(dataset):
    config = cfg.Config()
    data_dir = f"{config.data_path_tudo}/{dataset}"
    logger.info(f"loading dataset {dataset} from TU Dortmund data.")
    files = [
        file.replace(f"{dataset}_", "").replace(".txt", "")
        for file in os.listdir(data_dir)
        if file.split("_")[0] == dataset
    ]
    g_indicator = np.loadtxt(f"{data_dir}/{dataset}_graph_indicator.txt", delimiter=",")
    g_labels = np.loadtxt(
        f"{data_dir}/{dataset}_graph_labels.txt", delimiter=","
    ).tolist()

    # create helpers
    N = np.max(g_indicator).astype(int)
    n_nodes = g_indicator.shape[0]
    n2g_dict = {i: int(g_ind) - 1 for i, g_ind in enumerate(g_indicator.tolist())}

    edge_labels_bool = "edge_labels" in files
    node_labels_bool = "node_labels" in files
    if node_labels_bool:
        node_labels = open(f"{data_dir}/{dataset}_node_labels.txt", "r")
    if edge_labels_bool:
        edge_labels = open(f"{data_dir}/{dataset}_edge_labels.txt", "r")
    A = open(f"{data_dir}/{dataset}_A.txt", "r")

    node_idx = 0
    g_list = []
    for g_idx in tqdm(range(N)):
        g = nx.Graph()
        while n2g_dict[node_idx] == g_idx:
            if node_labels_bool:
                g.add_node(node_idx, lab=int(node_labels.readline().strip()))
            else:
                g.add_node(node_idx)
            node_idx += 1
            if node_idx == n_nodes:
                break

        edge = A.readline().strip().replace(" ", "").split(",")
        while (n2g_dict[int(edge[0]) - 1] == g_idx) & (edge != ""):
            if edge_labels_bool:
                g.add_edge(
                    int(edge[0]) - 1,
                    int(edge[1]) - 1,
                    lab=int(edge_labels.readline().strip()),
                )
            else:
                g.add_edge(int(edge[0]) - 1, int(edge[1]) - 1)
            edge = A.readline().strip().replace(" ", "").split(",")
            if edge[0] == "":
                break

        if not node_labels_bool:
            nx.set_node_attributes(g, dict(g.degree()), "lab")

        g_list.append(g)

    logger.info(f"# graphs -- {len(g_list)}")

    return g_list, g_labels


def load_graphs_duetal(dataset):
    logger.info(f"loading dataset {dataset} from Du et al. data.")

    if dataset in ["IMDBBINARY", "COLLAB", "IMDBMULTI"]:
        degree_as_label = True
    elif dataset in ["MUTAG", "PROTEINS", "PTC", "NCI1"]:
        degree_as_label = False

    config = cfg.Config()
    data_dir = f"{config.data_path_duetal}/{dataset}"

    g_list = []
    g_labels = []
    label_dict = {}
    feat_dict = {}

    with open(f"{data_dir}/{dataset}.txt", "r") as f:
        n_g = int(f.readline().strip())
        for i in tqdm(range(n_g)):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            n_edges = 0
            for j in range(n):
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    # attr = None
                else:
                    row = [int(w) for w in row[:tmp]]
                    # attr = np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                g.add_node(j, lab=feat_dict[row[0]])

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if degree_as_label:
                nx.set_node_attributes(g, dict(g.degree()), "lab")

            assert len(g) == n

            g_list.append(g)
            g_labels.append(label_dict[l])

    logger.info(f"# classes -- {len(label_dict)}")
    logger.info(f"# data -- {len(g_list)}")

    return g_list, g_labels


def load_gntk_matrices(source="TU_DO", dataset="MUTAG", min_scale_mat=0):
    config = cfg.Config()

    matrix_dir = f"{config.matrix_path}/GNTK_{source}/{dataset}"
    data_dir = f"{config.data_path}/{source}/{dataset}"
    kernel_matrices = {}

    matrix_list = [
        mat_name
        for mat_name in os.listdir(matrix_dir)
        if os.path.isdir(f"{matrix_dir}/{mat_name}")
    ]
    for mat_name in matrix_list:
        with open(f"{matrix_dir}/{mat_name}/gram.pkl", "rb") as f:
            mat = pickle.load(f)

        # if args.min_scale_mat is True, scale each matrix
        # by its min. Done by Du et al.
        # False by default due to potential leak of test data
        if min_scale_mat:
            mat = mat / mat.min()

        kernel_matrices["_".join(mat_name.split("_")[2:])] = mat

    labels = np.loadtxt(f"{data_dir}/{dataset}_graph_labels.txt")

    return (kernel_matrices, labels)



if __name__ == "__main__":
    logger = logging.get_logger()

    logger.info("test graph loader")
    tudo_graphs, tudo_labels = load_graphs("TU_DO", "MUTAG")
    duetal_graphs, duetal_labels = load_graphs("Du_et_al", "MUTAG")

    if len(tudo_graphs) == len(duetal_graphs):
        logger.info("Graphs loaded successfully")
    else:
        raise ValueError("Length of graph lists differs.")

    logger.info("test matrix loader")
    try:
        matrices, labels = load_gntk_matrices("TU_DO", "MUTAG", False)
        logger.info("Loading kernel matrices successful.")
        logger.info(f"N matrices -- {len(matrices)}")
    except:
        raise ValueError("Loading kernel matrices failed")

