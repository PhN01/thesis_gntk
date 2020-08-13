"""
@author: Philipp Nikolaus
@date: 17.12.2019
@references: Du, Simon et al. (2019): Graph Neural Tangent Kernel - Fusing Graph Neural Networks and Graph Kernels
"""


import numpy as np

from multiprocessing import Pool
from functools import partial
from itertools import combinations
from tqdm import tqdm

import networkx as nx


"""
General info:
This file contains an implementation of the graph neural tangent kernel by Du et al. (2019) based on the
implementation in gntk.py.

Here I implement the following updates:
1. solve sigma update as matrix operation (handwritten notes p.30/31)

Identified issues:
1. Redundancy of the following in each aggregation step
    kronecker product
    aggregation weight
    calculation of adjacency matrix
2. label matrices and adjacency matrices should be provided to gntk function, e.g. as lists
3. naming conventions partly inconsistent
4. entire sig_gxgx / sig_gygy stored, although only diagonal elements required for sigma updates
"""


def get_labelmatrix(graph):
    """
    TODO: provide label matrices as external gntk input
    """
    nodes = list(graph.nodes.data())
    lab = np.zeros((len(nodes), len(nodes[0][1]['onehot'])))
    for i in range(len(nodes)):
        lab[i, :] = nodes[i][1]['onehot']
    return lab


def calc_update(prev_sig_gxgx, prev_sig_gxgy, prev_sig_gygy, csig):
    c1c2_mat = np.matmul(
        np.sqrt(np.diag(prev_sig_gxgx))[:, np.newaxis],
        np.sqrt(np.diag(prev_sig_gygy))[np.newaxis, :]
    )
    lam = prev_sig_gxgy / c1c2_mat
    lam = np.minimum(np.maximum(lam,-1),1)

    sig_gxgy = (lam * (np.pi - np.arccos(lam)) + np.sqrt(1 - lam ** 2)) / (2 * np.pi) * c1c2_mat
    sig_gxgy = csig * sig_gxgy
    sigp_gxgy = (np.pi - np.arccos(lam)) / (2 * np.pi)
    sigp_gxgy = csig * sigp_gxgy

    return sig_gxgy, sigp_gxgy


def update_sig_gxgy(prev_sig_gxgx, prev_sig_gxgy, prev_sig_gygy, prev_theta, csig):
    sig_gxgy, sigp_gxgy = calc_update(prev_sig_gxgx, prev_sig_gxgy, prev_sig_gygy, csig)
    theta = prev_theta * sigp_gxgy + sig_gxgy
    return sig_gxgy, theta

def update_sig_gxgx(prev_sig_gxgx, csig):
    sig_gxgy, _ = calc_update(prev_sig_gxgx, prev_sig_gxgx, prev_sig_gxgx, csig)
    return sig_gxgy


def FC_layer(prev_sig_gxgx, prev_sig_gxgy, prev_sig_gygy, prev_theta, csig):
    sig_gxgy, theta = update_sig_gxgy(prev_sig_gxgx, prev_sig_gxgy, prev_sig_gygy, prev_theta, csig)
    sig_gxgx = update_sig_gxgx(prev_sig_gxgx, csig)
    sig_gygy = update_sig_gxgx(prev_sig_gygy, csig)

    return sig_gxgx, sig_gxgy, sig_gygy, theta


def AGG(graph0, graph1, sig_gxgx, sig_gxgy, sig_gygy, theta, cu_type):
    """
    TODO: resolve redundancy of adj_mat, kronecker product and weight matrix calculation
    """
    graph0_adj = np.minimum(nx.adjacency_matrix(graph0).toarray(), 1) + np.eye(sig_gxgx.shape[0])
    graph1_adj = np.minimum(nx.adjacency_matrix(graph1).toarray(), 1) + np.eye(sig_gygy.shape[0])

    # calculation of aggregation weight matrices
    if cu_type == "one":
        cu_mat = np.ones((graph0_adj.shape[0], graph1_adj.shape[0]))
        cu_mat_g0 = np.ones((graph0_adj.shape[0], graph0_adj.shape[0]))
        cu_mat_g1 = np.ones((graph1_adj.shape[0], graph1_adj.shape[0]))
    elif cu_type == "norm":
        cu_mat = 1 / np.matmul(np.sum(graph0_adj, axis=1)[:, np.newaxis], np.sum(graph1_adj, axis=0)[np.newaxis, :])
        cu_mat_g0 = 1 / np.matmul(np.sum(graph0_adj, axis=1)[:, np.newaxis], np.sum(graph0_adj, axis=0)[np.newaxis, :])
        cu_mat_g1 = 1 / np.matmul(np.sum(graph1_adj, axis=1)[:, np.newaxis], np.sum(graph1_adj, axis=0)[np.newaxis, :])

    adjk = np.kron(graph0_adj, graph1_adj)
    adjk_g0 = np.kron(graph0_adj, graph0_adj)
    adjk_g1 = np.kron(graph1_adj, graph1_adj)

    sig_gxgy_agg = cu_mat * np.matmul(adjk, sig_gxgy.reshape(-1)).reshape(sig_gxgy.shape)
    sig_gxgx_agg = cu_mat_g0 * np.matmul(adjk_g0, sig_gxgx.reshape(-1)).reshape(sig_gxgx.shape)
    sig_gygy_agg = cu_mat_g1 * np.matmul(adjk_g1, sig_gygy.reshape(-1)).reshape(sig_gygy.shape)
    theta_agg = cu_mat * np.matmul(adjk, theta.reshape(-1)).reshape(theta.shape)

    return sig_gxgx_agg, sig_gxgy_agg, sig_gygy_agg, theta_agg


# def store_blockitems(sig_gxgx, sig_gxgy, sig_gygy, theta):
#     res = {
#         'sl_gg': sig_gxgx,
#         'sl_gg_': sig_gxgy,
#         'sl_g_g_': sig_gygy,
#         'theta': theta
#     }
#     return res


def BLOCK(graph0, graph1, prev_sig_gxgx, prev_sig_gxgy, prev_sig_gygy, prev_theta, r, cu_type, csig):
    """putting together all block elements (aggregation and R FC layers)"""
    # block_items = {}
    sig_gxgx = prev_sig_gxgx
    sig_gxgy = prev_sig_gxgy
    sig_gygy = prev_sig_gygy
    theta = prev_theta
    # block_items['prev'] = store_blockitems(sig_gxgx, sig_gxgy, sig_gygy, theta)

    sig_gxgx, sig_gxgy, sig_gygy, theta = AGG(graph0, graph1, sig_gxgx, sig_gxgy, sig_gygy, theta,
                                                              cu_type)
    # block_items['agg'] = store_blockitems(sig_gxgx, sig_gxgy, sig_gygy, theta)
    for fc in range(r):
        sig_gxgx, sig_gxgy, sig_gygy, theta = FC_layer(sig_gxgx, sig_gxgy, sig_gygy, theta, csig)
        # block_items[fc] = store_blockitems(sig_gxgx, sig_gxgy, sig_gygy, theta)

    # return sig_gxgx, sig_gxgy, sig_gygy, theta, block_items
    return sig_gxgx, sig_gxgy, sig_gygy, theta


def READOUT(theta, jk=True):
    if not jk:
        return np.array(list(theta.values())[-1]).sum()
    if jk:
        return np.array(list(theta.values())).sum()


def gntk_pair(graph0, graph1, L, R, cu_type, csig):
    """
    TODO: provide adjacency matrices as external gntk input
    """
    g0_lab = get_labelmatrix(graph0)
    g1_lab = get_labelmatrix(graph1)

    sig_gxgx = np.matmul(g0_lab, g0_lab.transpose())
    sig_gxgy = np.matmul(g0_lab, g1_lab.transpose())
    sig_gygy = np.matmul(g1_lab, g1_lab.transpose())
    theta_dict = {0: sig_gxgy.copy()}
    theta = sig_gxgy.copy()
    # gntk_items = {}
    # gntk_items['start'] = store_blockitems(sig_gxgx, sig_gxgy, sig_gygy, theta)

    for block in range(1, L + 1):
        # sig_gxgx, sig_gxgy, sig_gygy, theta, gntk_items[block] = BLOCK(graph0, graph1, sig_gxgx, sig_gxgy, sig_gygy, theta, R, cu_type, csig)
        sig_gxgx, sig_gxgy, sig_gygy, theta = BLOCK(graph0, graph1, sig_gxgx, sig_gxgy, sig_gygy, theta, R, cu_type, csig)
        theta_dict[block] = theta.copy()
    theta_L = READOUT(theta_dict)

    # return theta_L, gntk_items
    return theta_L

def gntk_pool_fun(pair, graph_list, L, R, cu_type, csig):
    graph0 = graph_list[pair[0]]
    graph1 = graph_list[pair[1]]
    k_gxgy = gntk_pair(graph0, graph1, L, R, cu_type,csig)
    return k_gxgy


from itertools import product
def gntk_gram(graph_list, L, R, cu_type, csig, ncores=1):
    ngraphs = len(graph_list)
    pair_list = [pair for pair in combinations(range(ngraphs),2)]
    pair_list += [(i,i) for i in range(ngraphs)]
    fun = partial(gntk_pool_fun,
                  graph_list=graph_list,
                  L=L,
                  R=R,
                  cu_type=cu_type,
                  csig=csig)
    with Pool(ncores) as pool:
        gram_list = list(tqdm(pool.imap(fun, pair_list), total=len(pair_list)))
        gram_mat = np.zeros((ngraphs, ngraphs))
        for i in range(len(gram_list)):
            if pair_list[i][0] == pair_list[i][1]:
                gram_mat[pair_list[i][0], pair_list[i][0]] = gram_list[i]
            else:
                gram_mat[pair_list[i][0], pair_list[i][1]] = gram_list[i]
                gram_mat[pair_list[i][1], pair_list[i][0]] = gram_list[i]
    return gram_mat


def log_result(result):
    global gram_list
    global pair_list
    gram_list.append(result)
    if len(gram_list) % (len(pair_list) // 10) == 0:
        print('\r{:.0%} done'.format(len(gram_list) / len(pair_list)), end=" ", flush=True)


# def gntk_gram(graph_list, L, R, cu_type, csig, ncores=1):
#     ngraphs = len(graph_list)
#     pair_list = [pair for pair in combinations(range(ngraphs),2)]
#     fun = partial(pair,
#                   graph_list=graph_list,
#                   L=L,
#                   R=R,
#                   cu_type=cu_type,
#                   csig=csig)
#     with Pool(ncores) as pool:
#         gram_list = []
#         for pair in pair_list:
#             gram_list = pool.apply_async(gntk_pool_fun, args=[pair,graph_list,L,R,cu_type,csig], callback)
#         gram_list = pool.map()
#             map(fun, pair_list)
#         gram_mat = np.zeros((ngraphs, ngraphs))
#         for i in range(len(gram_list)):
#             gram_mat[pair_list[i][0], pair_list[i][1]] = gram_list[i]
#     return gram_mat