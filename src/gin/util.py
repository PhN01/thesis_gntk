import networkx as nx
import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedKFold
import logging

from tqdm import tqdm
import os

import sys

sys.path.append(".")

from src.config import config as cfg

logger = logging.getLogger(__name__)


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        """
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        """
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0


def load_data(dataset, degree_as_tag):
    """
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    """

    print("loading data")
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open("dataset/%s/%s.txt" % (dataset, dataset), "r") as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = (
                        [int(w) for w in row[:tmp]],
                        np.array([float(w) for w in row[tmp:]]),
                    )
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    # add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    # Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[
            range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]
        ] = 1

    print("# classes: %d" % len(label_dict))
    print("# maximum node tag: %d" % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)


def load_data_tudo(dataset):
    """
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    """
    config = cfg.Config()

    data_dir = config.data_path_tudo

    if dataset in ["IMDBBINARY", "COLLAB", "IMDBMULTI"]:
        degree_as_tag = True
    elif dataset in ["MUTAG", "PROTEINS", "PTC", "NCI1"]:
        degree_as_tag = False

    logger.info("Loading data")
    g_list = []
    label_dict = {}
    feat_dict = {}

    files = [
        file.replace("{}_".format(dataset), "").replace(".txt", "")
        for file in os.listdir(os.path.join(data_dir, dataset))
        if file.split("_")[0] == dataset
    ]
    g_indicator = np.loadtxt(
        os.path.join(data_dir, dataset, "{}_graph_indicator.txt".format(dataset)),
        delimiter=",",
    )
    g_labels = np.loadtxt(
        os.path.join(data_dir, dataset, "{}_graph_labels.txt".format(dataset)),
        delimiter=",",
    ).tolist()

    # create helpers
    n_g = np.max(g_indicator).astype(int)
    n_nodes = g_indicator.shape[0]
    n2g_dict = {i: int(g_ind) - 1 for i, g_ind in enumerate(g_indicator.tolist())}

    edge_labels_bool = "edge_labels" in files
    node_labels_bool = "node_labels" in files
    if node_labels_bool:
        node_labels = open(
            os.path.join(data_dir, dataset, "{}_node_labels.txt".format(dataset)), "r"
        )
    if edge_labels_bool:
        edge_labels = open(
            os.path.join(data_dir, dataset, "{}_edge_labels.txt".format(dataset)), "r"
        )
    A = open(os.path.join(data_dir, dataset, "{}_A.txt".format(dataset)), "r")

    node_idx = 0
    for g_idx in tqdm(range(n_g)):
        if not g_labels[g_idx] in label_dict:
            mapped = len(label_dict)
            label_dict[g_labels[g_idx]] = mapped

        g = nx.Graph()
        g_node_idx = 0
        node_dict = {}
        node_tags = []
        while n2g_dict[node_idx] == g_idx:
            node_dict[node_idx] = g_node_idx
            if node_labels_bool:
                l = int(node_labels.readline().strip())
                if not l in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[l] = mapped
                g.add_node(g_node_idx)
                node_tags.append(feat_dict[l])
            node_idx += 1
            g_node_idx += 1
            if node_idx == n_nodes:
                break

        edge = A.readline().strip().replace(" ", "").split(",")
        while (n2g_dict[int(edge[0]) - 1] == g_idx) & (edge != ""):
            v1 = int(edge[0]) - 1
            v2 = int(edge[1]) - 1
            g.add_edge(node_dict[v1], node_dict[v2])
            edge = A.readline().strip().replace(" ", "").split(",")
            if edge[0] == "":
                break

        g_list.append(S2VGraph(g, label_dict[g_labels[g_idx]], node_tags))
        inverse_label_dict = {v: k for k, v in label_dict.items()}

    # add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for _ in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        # g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    # Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[
            range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]
        ] = 1

    logger.info("# classes: %d" % len(label_dict))
    logger.info("# maximum node tag: %d" % len(tagset))
    logger.info("# data: %d" % len(g_list))

    return g_list, len(label_dict), g_labels, inverse_label_dict


def separate_data(dataset, graph_list, seed, fold_idx, g_labels):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    # if dataset in ['IMDBMULTI']:
    #     label_dict = {0: 1, 1: 2, 2: 3}
    # else:
    #     label_dict = {0: -1, 1: 1}
    #
    # labels = [label_dict[graph.label] for graph in graph_list]

    train_folds = []
    test_folds = []
    for fold_index, (train_index, test_index) in enumerate(
        skf.split(np.arange(len(graph_list)), g_labels)
    ):
        train_folds.append(train_index.tolist())
        test_folds.append(test_index.tolist())
    train_idx = train_folds[fold_idx]
    test_idx = test_folds[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list, train_idx, test_idx
