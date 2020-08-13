import numpy as np
import scipy.sparse as sparse

import networkx as nx
import igraph as ig


def node_lab_mat(g_list):
    lab_lists = [[node[1]["lab"] for node in graph.nodes.data()] for graph in g_list]
    lab_set = np.unique(
        [item[j] for item in lab_lists for j in range(len(item))]
    ).tolist()
    lab_dict = {old: new for new, old in enumerate(lab_set)}

    mat_list = []
    for g in g_list:
        lab_mat = np.zeros((len(g), len(lab_set)))
        labels = [lab_dict[node[1]["lab"]] for node in g.nodes.data()]
        lab_mat[range(len(g)), labels] = 1
        mat_list.append(lab_mat)
    return mat_list


def get_graph_adj(graph):
    adj = nx.adjacency_matrix(graph).toarray() + np.eye(len(graph.nodes.data()))
    adj_sparse = sparse.csr_matrix(adj)
    return adj_sparse


def conv_graph_nx2ig(nx_graph_list):
    ig_graph_list = []
    for graph in nx_graph_list:
        ig_graph = ig.Graph()
        for vertex in graph.nodes.data():
            ig_graph.add_vertex(str(vertex[0]), label=vertex[1]["lab"])
        for edge in graph.edges.data():
            try:
                ig_graph.add_edge(str(edge[0]), str(edge[1]), label=edge[2]["lab"])
            except:
                ig_graph.add_edge(str(edge[0]), str(edge[1]))
        ig_graph_list.append(ig_graph)
    return ig_graph_list


def get_data_origin(data_path):
    if "TU_DO" in data_path:
        return "TU_DO"
    elif "Du_et_al" in data_path:
        return "Du_et_al"
    else:
        raise ValueError(
            "Unknown data source. One of ['TU_DO','Du_et_al'] must be in path."
        )