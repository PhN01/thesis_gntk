"""
@author: Philipp Nikolaus
@date: 27.01.2020
@references: Du, Simon et al. (2019): Graph Neural Tangent Kernel - Fusing Graph Neural Networks and Graph Kernels
"""


import numpy as np
import scipy.sparse as sparse
from itertools import combinations
from tqdm import tqdm
import joblib

"""
General info:
This file contains an implementation of the graph neural tangent kernel by Du et al. (2019) based on the
implementation in gntk_v1_1.py.

Here I implement the following updates:
1. Removal of redundant parallelization methods for gram calculation
2. Implemented ability to specify higher degree activation function
3. csig parameter hard coded
    
Identified issues:
1. entire sig_gxgx / sig_gygy stored, although only diagonal elements required for sigma updates
    potentially problematic for large datasets. keep for now, unless running into problems.
2. documentation
3. be more consistent with the use of sparse matrices.
4. Potentially remove higher degree activation code
"""

def calc_update(sig, sig_g0, sig_g1):
    c1c2_mat = np.matmul(
        np.sqrt(np.diag(sig_g0))[:, np.newaxis],
        np.sqrt(np.diag(sig_g1))[np.newaxis, :]
    )
    lam = sig / c1c2_mat
    lam = np.minimum(np.maximum(lam,-1),1)
    # activation degree = 1
    sig = (lam * (np.pi - np.arccos(lam)) + np.sqrt(1 - lam ** 2)) / (2 * np.pi) * c1c2_mat
    sig = 2 * sig
    sig_dot = (np.pi - np.arccos(lam)) / (2 * np.pi)
    sig_dot = 2 * sig_dot
    # activation degree = 2
    # sig = (3 * np.sqrt(1 - lam ** 2) * lam + (np.pi - np.arccos(lam)) * (1 + np.cos(lam))) / (2 * np.pi) * c1c2_mat
    # sig = 2 * sig
    # sig_dot = (lam * (np.pi - np.arccos(lam)) + np.sqrt(1 - lam ** 2)) / (2 * np.pi)
    # sig_dot = 2 * sig_dot

    return sig, sig_dot


def update_sig(sig, sig_g0, sig_g1, theta):
    sig, sig_dot = calc_update(sig, sig_g0, sig_g1)
    theta = theta * sig_dot + sig
    return sig, theta

def update_sig_same(sig):
    sig, _ = calc_update(sig, sig, sig)
    return sig


def FC_layer(sig, sig_g0, sig_g1, theta):
    sig, theta = update_sig(sig, sig_g0, sig_g1, theta)
    sig_g0 = update_sig_same(sig_g0)
    sig_g1 = update_sig_same(sig_g1)

    return sig, sig_g0, sig_g1, theta


def AGG(adj_kprod, adj_kprod_g0, adj_kprod_g1,
        sig, sig_g0, sig_g1, theta,
        scale_mat, scale_mat_g0, scale_mat_g1):

    sig = scale_mat * sparse.csr_matrix.dot(adj_kprod, sig.reshape(-1)).reshape(sig.shape)
    sig_g0 = scale_mat_g0 * sparse.csr_matrix.dot(adj_kprod_g0, sig_g0.reshape(-1)).reshape(sig_g0.shape)
    sig_g1 = scale_mat_g1 * sparse.csr_matrix.dot(adj_kprod_g1, sig_g1.reshape(-1)).reshape(sig_g1.shape)
    theta = scale_mat * sparse.csr_matrix.dot(adj_kprod, theta.reshape(-1)).reshape(theta.shape)

    return sig, sig_g0, sig_g1, theta


def READOUT(theta, jk=True):
    if not jk:
        return np.array(theta[-1]).sum()
    if jk:
        return np.array(theta).sum()


def gntk_pair(g0_lab, g1_lab, g0_adj, g1_adj, L, R, scale, jk):
    # calculation of aggregation weight matrices
    if scale == "uniform":
        scale_mat = np.ones((g0_adj.shape[0], g1_adj.shape[0]))
        scale_mat_g0 = np.ones((g0_adj.shape[0], g0_adj.shape[0]))
        scale_mat_g1 = np.ones((g1_adj.shape[0], g1_adj.shape[0]))
    elif scale == "degree":
        scale_mat = 1 / np.matmul(
            np.sum(g0_adj.toarray(), axis=1)[:, np.newaxis],
            np.sum(g1_adj.toarray(), axis=0)[np.newaxis, :]
        )
        scale_mat_g0 = 1 / np.matmul(
            np.sum(g0_adj.toarray(), axis=1)[:, np.newaxis],
            np.sum(g0_adj.toarray(), axis=0)[np.newaxis, :]
        )
        scale_mat_g1 = 1 / np.matmul(
            np.sum(g1_adj.toarray(), axis=1)[:, np.newaxis],
            np.sum(g1_adj.toarray(), axis=0)[np.newaxis, :]
        )

    adj_kprod = sparse.kron(g0_adj, g1_adj)
    adj_kprod_g0 = sparse.kron(g0_adj, g0_adj)
    adj_kprod_g1 = sparse.kron(g1_adj, g1_adj)

    sig = np.matmul(g0_lab, g1_lab.transpose())
    sig_g0 = np.matmul(g0_lab, g0_lab.transpose())
    sig_g1 = np.matmul(g1_lab, g1_lab.transpose())
    theta = sig.copy()
    theta_jk = []
    theta_jk.append(sig.copy())

    for block in range(1, L + 1):
        sig, sig_g0, sig_g1, theta = AGG(
            adj_kprod, adj_kprod_g0, adj_kprod_g1,
            sig, sig_g0, sig_g1, theta,
            scale_mat, scale_mat_g0, scale_mat_g1
        )
        for fc in range(R):
            sig, sig_g0, sig_g1, theta = FC_layer(sig, sig_g0, sig_g1, theta)
        theta_jk.append(theta.copy())
    theta = READOUT(theta_jk, jk)
    return theta


def gntk_gram_joblib(lab_list, adj_list, L, R, scale, jk, n_jobs):
    ngraphs = len(lab_list)
    pair_list = [(i, i) for i in range(ngraphs)]
    pair_list += [pair for pair in combinations(range(ngraphs), 2)]

    results = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(gntk_pair)(
        lab_list[pair[0]],lab_list[pair[1]],
        adj_list[pair[0]],adj_list[pair[1]],
        L, R, norm, jk
    ) for pair in tqdm(pair_list))

    gram_mat = np.zeros((ngraphs, ngraphs))
    for i, val in enumerate(results):
        pair = pair_list[i]
        gram_mat[pair[0], pair[1]] = val
        gram_mat[pair[1], pair[0]] = val
    return gram_mat


def gntk_gram(lab_list, adj_list, L, R, scale, jk):
    ngraphs = len(lab_list)
    diag_pair_list = [(i, i) for i in range(ngraphs)]
    pair_list = [pair for pair in combinations(range(ngraphs),2)]
    gram_mat = np.zeros((ngraphs, ngraphs))

    diag_gram_items = (gntk_pair(
        lab_list[pair[0]], lab_list[pair[1]],
        adj_list[pair[0]], adj_list[pair[1]],
        L, R, cu_type, jk
    ) for pair in diag_pair_list)
    gram_items = (gntk_pair(
        lab_list[pair[0]], lab_list[pair[1]],
        adj_list[pair[0]], adj_list[pair[1]],
        L, R, cu_type, jk
    ) for pair in pair_list)

    for i, item in enumerate(diag_gram_items):
        gram_mat[i,i] = item
    for i, item in enumerate(gram_items):
        gram_mat[pair_list[i][0], pair_list[i][1]] = item
        gram_mat[pair_list[i][1], pair_list[i][0]] = item
        print("\r{}/{}".format(i, len(pair_list)), end='', flush=True)
    return gram_mat