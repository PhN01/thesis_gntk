"""
@author: Philipp Nikolaus
@date: 05.12.2019
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
This file contains an implementation of the graph neural tangent kernel by Du et al. (2019) based on the 'literal'
implementation in gntk.py.

Here I implement the following updates:
1. calculate sig0 directly as a matrix product
2. remove intermediate Auv calculation
"""


def get_labelmatrix(graph):
    nodes = list(graph.nodes.data())
    lab = np.zeros((len(nodes), len(nodes[0][1]['onehot'])))
    for i in range(len(nodes)):
        lab[i, :] = nodes[i][1]['onehot']
    return lab


def calc_update(prev_sig_uu, prev_sig_uv, prev_sig_vv, csig):
    c1 = np.sqrt(prev_sig_uu)
    c2 = np.sqrt(prev_sig_vv)
    lam = np.minimum(np.maximum(prev_sig_uv/c1/c2,-1),1)
    sig_item = csig * (lam * (np.pi - np.arccos(lam)) + np.sqrt(1 - lam**2)) / (2 * np.pi) * c1 * c2
    sigp_item = csig * (np.pi - np.arccos(lam)) / (2 * np.pi)
    return sig_item, sigp_item


def update_item(prev_sig_uu, prev_sig_uv, prev_sig_vv, prev_theta_uv, csig):
    sig_uv, sigp_uv = calc_update(prev_sig_uu, prev_sig_uv, prev_sig_vv, csig)
    theta_uv = prev_theta_uv * sigp_uv + sig_uv
    return sig_uv, theta_uv


def FC_layer(prev_sig_gxgx, prev_sig_gxgy, prev_sig_gygy, prev_theta, csig):
    sig_gxgy = np.zeros((prev_sig_gxgy.shape[0], prev_sig_gxgy.shape[1]))
    theta = np.zeros((prev_theta.shape[0], prev_theta.shape[1]))

    sig_gxgx = np.zeros((prev_sig_gxgx.shape[0], prev_sig_gxgx.shape[0]))
    sig_gygy = np.zeros((prev_sig_gygy.shape[0], prev_sig_gygy.shape[0]))

    for i in range(prev_theta.shape[0]):
        for j in range(prev_theta.shape[1]):
            sig_gxgy[i,j], theta[i,j] = update_item(
                prev_sig_gxgx[i,i],
                prev_sig_gxgy[i,j],
                prev_sig_gygy[j,j],
                prev_theta[i,j],
                csig
             )

    for i in range(prev_sig_gxgx.shape[0]):
        for j in range(prev_sig_gxgx.shape[0]):
            sig_gxgx[i,j], _ = update_item(
                prev_sig_gxgx[i,i],
                prev_sig_gxgx[i,j],
                prev_sig_gxgx[j,j],
                prev_sig_gxgx[i,j],
                csig
            )

    for i in range(prev_sig_gygy.shape[1]):
        for j in range(prev_sig_gygy.shape[1]):
            sig_gygy[i,j], _ = update_item(
                prev_sig_gygy[i,i],
                prev_sig_gygy[i,j],
                prev_sig_gygy[j,j],
                prev_sig_gygy[i,j],
                csig
            )

    return sig_gxgx, sig_gxgy, sig_gygy, theta


def f_cu(adj, cu_type):

    """
    For normalization case
    TODO: important insight: weights c_gx * c_gy could be calculated directly as matrix
    reduce adjacency matrices to vectors by summing across one axis and multiply vectors such
    that product is of form (n0 x n1)
    see handwritten notes p. 6/7
    """
    if cu_type=="one":
        return 1
    if cu_type=="norm":
        return 1/np.sum(adj)


def f_agg_el(adj0, adj1, mat, cu_type):
    g0_ind = np.array(adj0).nonzero()[0]
    g1_ind = np.array(adj1).nonzero()[0]
    sig_agg_el = 0
    for i in g0_ind:
        for j in g1_ind:
            sig_agg_el+=mat[i,j]
    cu = f_cu(adj0, cu_type)
    cv = f_cu(adj1, cu_type)
    return cu*cv*sig_agg_el


def AGG(graph0, graph1, sig_gxgx, sig_gxgy, sig_gygy, theta, cu_type):
    """
    TODO: entire matrix level aggregation could potentially be done using the kronecker product
    https://www.inference.vc/dilated-convolutions-and-kronecker-factorisation/
    https://arxiv.org/abs/1512.09194
    """
    sig_gxgx_agg = np.zeros((sig_gxgx.shape))
    sig_gxgy_agg = np.zeros((sig_gxgy.shape))
    sig_gygy_agg = np.zeros((sig_gygy.shape))
    theta_agg = np.zeros((theta.shape))
    graph0_adj = np.minimum(nx.adjacency_matrix(graph0).toarray(),1)+np.eye(sig_gxgx.shape[0])
    graph1_adj = np.minimum(nx.adjacency_matrix(graph1).toarray(),1)+np.eye(sig_gygy.shape[0])
    for i in range(sig_gxgy.shape[0]):
        for j in range(sig_gxgy.shape[1]):
            adj0 = graph0_adj[:,i]
            adj1 = graph1_adj[:,j]
            sig_gxgy_agg[i,j] = f_agg_el(adj0, adj1, sig_gxgy, cu_type)
            theta_agg[i,j] = f_agg_el(adj0, adj1, theta, cu_type)
    for i in range(sig_gxgx.shape[0]):
        for j in range(sig_gxgx.shape[0]):
            adj0 = graph0_adj[:, i]
            adj1 = graph0_adj[:, j]
            sig_gxgx_agg[i,j] = f_agg_el(adj0, adj1, sig_gxgx, cu_type)
    for i in range(sig_gygy.shape[0]):
        for j in range(sig_gygy.shape[0]):
            adj0 = graph1_adj[:, i]
            adj1 = graph1_adj[:, j]
            sig_gygy_agg[i,j] = f_agg_el(adj0, adj1, sig_gygy, cu_type)

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