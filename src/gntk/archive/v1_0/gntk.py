"""
@author: Philipp Nikolaus
@date: 11.01.2020
@references: Du, Simon et al. (2019): Graph Neural Tangent Kernel - Fusing Graph Neural Networks and Graph Kernels
"""


import numpy as np

from multiprocessing import Pool
from functools import partial
from itertools import combinations
from tqdm import tqdm


"""
General info:
This file contains an implementation of the graph neural tangent kernel by Du et al. (2019) based on the
implementation in gntk_v0_4.py.

Here I implement the following updates:
1. Consistency of naming conventions
2. general cleanup

Identified issues:
1. entire sig_gxgx / sig_gygy stored, although only diagonal elements required for sigma updates
    potentially problematic for large datasets. keep for now, unless running into problems.
2. documentation
3. the Kronecker product outputs a n^2 x n^2 matrix. This leads to memory issues for large graphs.
    Try to use sparse matrices
"""

def calc_update(sig, sig_g0, sig_g1, csig):
    c1c2_mat = np.matmul(
        np.sqrt(np.diag(sig_g0))[:, np.newaxis],
        np.sqrt(np.diag(sig_g1))[np.newaxis, :]
    )
    lam = sig / c1c2_mat
    lam = np.minimum(np.maximum(lam,-1),1)

    sig = (lam * (np.pi - np.arccos(lam)) + np.sqrt(1 - lam ** 2)) / (2 * np.pi) * c1c2_mat
    sig = csig * sig
    sig_dot = (np.pi - np.arccos(lam)) / (2 * np.pi)
    sig_dot = csig * sig_dot

    return sig, sig_dot


def update_sig(sig, sig_g0, sig_g1, theta, csig):
    sig, sig_dot = calc_update(sig, sig_g0, sig_g1, csig)
    theta = theta * sig_dot + sig
    return sig, theta

def update_sig_same(sig, csig):
    sig, _ = calc_update(sig, sig, sig, csig)
    return sig


def FC_layer(sig, sig_g0, sig_g1, theta, csig):
    sig, theta = update_sig(sig, sig_g0, sig_g1, theta, csig)
    sig_g0 = update_sig_same(sig_g0, csig)
    sig_g1 = update_sig_same(sig_g1, csig)

    return sig, sig_g0, sig_g1, theta


def AGG(adjk, adjk_g0, adjk_g1, sig, sig_g0, sig_g1, theta, cu_mat, cu_mat_g0, cu_mat_g1):

    sig = cu_mat * np.matmul(adjk, sig.reshape(-1)).reshape(sig.shape)
    sig_g0 = cu_mat_g0 * np.matmul(adjk_g0, sig_g0.reshape(-1)).reshape(sig_g0.shape)
    sig_g1 = cu_mat_g1 * np.matmul(adjk_g1, sig_g1.reshape(-1)).reshape(sig_g1.shape)
    theta = cu_mat * np.matmul(adjk, theta.reshape(-1)).reshape(theta.shape)

    return sig, sig_g0, sig_g1, theta


def BLOCK(adjk, adjk_g0, adjk_g1, sig, sig_g0, sig_g1, theta, r, cu_mat, cu_mat_g0, cu_mat_g1, csig):
    """putting together all block elements (aggregation and R FC layers)"""
    sig, sig_g0, sig_g1, theta = AGG(adjk, adjk_g0, adjk_g1, sig, sig_g0, sig_g1, theta, cu_mat, cu_mat_g0, cu_mat_g1)
    for fc in range(r):
        sig, sig_g0, sig_g1, theta = FC_layer(sig, sig_g0, sig_g1, theta, csig)
    return sig, sig_g0, sig_g1, theta


def READOUT(theta, jk=True):
    if not jk:
        return np.array(theta[-1]).sum()
    if jk:
        return np.array(theta).sum()


def gntk_pair(g0_lab, g1_lab, g0_adj, g1_adj, L, R, cu_type, csig, jk):
    # calculation of aggregation weight matrices
    if cu_type == "one":
        cu_mat = np.ones((g0_adj.shape[0], g1_adj.shape[0]))
        cu_mat_g0 = np.ones((g0_adj.shape[0], g0_adj.shape[0]))
        cu_mat_g1 = np.ones((g1_adj.shape[0], g1_adj.shape[0]))
    elif cu_type == "norm":
        cu_mat = 1 / np.matmul(np.sum(g0_adj, axis=1)[:, np.newaxis], np.sum(g1_adj, axis=0)[np.newaxis, :])
        cu_mat_g0 = 1 / np.matmul(np.sum(g0_adj, axis=1)[:, np.newaxis], np.sum(g0_adj, axis=0)[np.newaxis, :])
        cu_mat_g1 = 1 / np.matmul(np.sum(g1_adj, axis=1)[:, np.newaxis], np.sum(g1_adj, axis=0)[np.newaxis, :])

    adjk = np.kron(g0_adj, g1_adj)
    adjk_g0 = np.kron(g0_adj, g0_adj)
    adjk_g1 = np.kron(g1_adj, g1_adj)

    sig = np.matmul(g0_lab, g1_lab.transpose())
    sig_g0 = np.matmul(g0_lab, g0_lab.transpose())
    sig_g1 = np.matmul(g1_lab, g1_lab.transpose())
    theta = sig.copy()
    theta_jk = []
    theta_jk.append(sig.copy())

    for block in range(1, L + 1):
        sig, sig_g0, sig_g1, theta = BLOCK(adjk, adjk_g0, adjk_g1, sig, sig_g0, sig_g1, theta, R, cu_mat, cu_mat_g0, cu_mat_g1, csig)
        theta_jk.append(theta.copy())
    theta = READOUT(theta_jk, jk)
    return theta

def gntk_pool_fun(pair, lab_list, adj_list, L, R, cu_type, csig, jk):
    g0_lab = lab_list[pair[0]]
    g1_lab = lab_list[pair[1]]
    g0_adj = adj_list[pair[0]]
    g1_adj = adj_list[pair[1]]
    k = gntk_pair(g0_lab, g1_lab, g0_adj, g1_adj, L, R, cu_type,csig, jk)
    return k


def gntk_gram_mp(lab_list, adj_list, L, R, cu_type, csig, jk, nc=1):
    ngraphs = len(lab_list)
    pair_list = [(i,i) for i in range(ngraphs)]
    pair_list += [pair for pair in combinations(range(ngraphs),2)]
    fun = partial(gntk_pool_fun,
                  lab_list=lab_list,
                  adj_list=adj_list,
                  L=L,
                  R=R,
                  cu_type=cu_type,
                  csig=csig,
                  jk=jk)
    with Pool(nc) as pool:
        gram_list = list(tqdm(pool.imap(fun, pair_list), total=len(pair_list)))
    gram_mat = np.zeros((ngraphs, ngraphs))
    for i in range(ngraphs):
        gram_mat[i,i] = gram_list[i]
    for i in range(ngraphs,len(pair_list)):
        val = gram_list[i]
        gram_mat[pair_list[i][0], pair_list[i][1]] = val
        gram_mat[pair_list[i][1], pair_list[i][0]] = val
    return gram_mat


def gntk_gram(lab_list, adj_list, L, R, cu_type, csig, jk):
    ngraphs = len(lab_list)
    diag_pair_list = [(i, i) for i in range(ngraphs)]
    pair_list = [pair for pair in combinations(range(ngraphs),2)]
    gram_mat = np.zeros((ngraphs, ngraphs))
    print('building kernel iterators')
    print('diag')
    diag_gram_items = (gntk_pair(lab_list[pair[0]], lab_list[pair[1]], adj_list[pair[0]], adj_list[pair[1]], L, R, cu_type, csig, jk) for pair in tqdm(diag_pair_list))
    print('off-diag')
    gram_items = (gntk_pair(lab_list[pair[0]], lab_list[pair[1]], adj_list[pair[0]], adj_list[pair[1]], L, R, cu_type, csig, jk) for pair in tqdm(pair_list))
    for i, item in tqdm(enumerate(diag_gram_items)):
        gram_mat[i,i] = item
    for i, item in tqdm(gram_items):
        gram_mat[pair_list[i][0], pair_list[i][1]] = item
        gram_mat[pair_list[i][1], pair_list[i][0]] = item
    return gram_mat