import sys

sys.path.append(".")
import argparse
import os

import pandas as pd
import numpy as np
from collections import Counter
import networkx as nx

from tqdm import tqdm

from src.utils import utils
from src.utils import logging
from src.data import data_loaders


def avg_edges(graph_list):
    """
    :param graph_list: list containing all graphs of a given dataset
    :return: returns the average number of edges of the graphs in {graph_list}
    """
    ne_list = [len(g.edges.data()) for g in graph_list]
    return np.mean(ne_list)


def avg_nodes(graph_list):
    """
    :param graph_list: list containing all graphs of a given dataset
    :return: returns the average number of nodes of the graphs in {graph_list}
    """
    nv_list = [len(g) for g in graph_list]
    return np.mean(nv_list)


def g_density(graph):
    """
    :param graph: an individual graph in nx format
    :return: returns the density of {graph}
    """
    ne = len(graph.edges.data())
    nv = len(graph)
    return 2 * ne / (nv * (nv - 1))


def avg_density(graph_list):
    """
    :param graph_list: list containing all graphs of a given dataset
    :return: returns the average density of the graphs in {graph_list}
    """
    density_list = [g_density(g) for g in graph_list]
    return np.mean(density_list)


def max_nodes(graph_list):
    """
    :param graph_list: list containing all graphs of a given dataset
    :return: returns the maxmimum number of nodes of the graphs in {graph_list}
    """
    nv_list = [len(g) for g in graph_list]
    return np.max(nv_list)


def avg_degree(graph_list):
    """
    :param graph_list: list containing all graphs of a given dataset
    :return: returns the average degree of all nodes of all graphs in {graph_list}
    """
    degree_list = [np.mean(list(dict(nx.degree(g)).values())) for g in graph_list]
    return np.mean(degree_list)


def avg_max_degree(graph_list):
    """
    :param graph_list: list containing all graphs of a given dataset
    :return: returns the average degree of all nodes of all graphs in {graph_list}
    """
    degree_list = [np.max(list(dict(nx.degree(g)).values())) for g in graph_list]
    return np.mean(degree_list)


def max_degree(graph_list):
    """
    :param graph_list: list containing all graphs of a given dataset
    :return: returns the average degree of all nodes of all graphs in {graph_list}
    """
    degree_list = [np.max(list(dict(nx.degree(g)).values())) for g in graph_list]
    return np.max(degree_list)


def class_ratio(labels):
    """
    :param labels: iterable containing the graph labels of a given dataset
    :return: returns a string of the form '{n_class0}:{n_class1}:...'
    """
    counts = Counter(labels)
    ratio_string = ":".join([str(count) for count in counts.values()])
    return ratio_string


def data_stats(dataset, graph_list, labels):
    """
    :param dataset: name of the dataset
    :param graph_list: list containing all graphs of the dataset
    :param labels: iterable containing the graph labels of a given dataset
    :return: returns a dictionary containing all relevant statistics of the dataset
    """
    stats = {
        "dataset": dataset,
        "n_graphs": len(graph_list),
        "classes": len(set(labels)),
        "density": avg_density(graph_list),
        "avg_nodes": avg_nodes(graph_list),
        "avg_edges": avg_edges(graph_list),
        "max_nodes": max_nodes(graph_list),
        "avg_degree": avg_degree(graph_list),
        "avg_max_degree": avg_max_degree(graph_list),
        "max_degree": max_degree(graph_list),
        "class_ratio": class_ratio(labels),
    }
    return stats


if __name__ == "__main__":
    logger = logging.get_logger()

    out_dir = "./reporting"
    utils.make_dirs_checked(out_dir)

    # create a list of all datasets in {args.data_dir}

    datasets = ["MUTAG", "PTC", "IMDBBINARY", "IMDBMULTI", "PROTEINS", "NCI1", "COLLAB"]
    logger.info(f"Computing summary statistics for datasets {datasets}")
    # empty list to iteratively append the dictionaries containing dataset
    # stats to
    stats_list = []

    # iterating over all datasets
    for dataset in tqdm(datasets):

        # load datasets
        graphs, labels = data_loaders.load_graphs_tudortmund(dataset)
        # apply data_stats() to the current dataset and append
        # the resulting dictionary to {stats_list}
        stats_list.append(data_stats(dataset, graphs, labels))

    # convert the list of dictionaries to a dataframe. by default
    # panda creates one row for each dictionary / dataset
    stats_df = pd.DataFrame(stats_list)

    logger.info("Summary statistics -- ")
    logger.info(f"{stats_df.to_string}")

    # store results to {args.data_dir/data_statistics.csv}
    logger.info(f"Storing results in {out_dir}/data_statistics.csv")
    stats_df.to_csv(os.path.join(out_dir, "data_statistics.csv"), sep=",", index=False)
