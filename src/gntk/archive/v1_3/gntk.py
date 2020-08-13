"""
@author: Philipp Nikolaus
@date: 20.02.2020
@references: Du, Simon et al. (2019): Graph Neural Tangent Kernel - Fusing Graph Neural Networks and Graph Kernels
"""

import numpy as np
import scipy.sparse as sparse

from itertools import combinations
import joblib

from tqdm import tqdm

"""
General info:
This file contains an implementation of the graph neural tangent kernel by Du et al. (2019) based on the
implementation in gntk_v1_1.py.

Here I implement the following updates:
1. Store only diagonals of sig_gxgx and sig_gygy to reduce memory
    
Identified issues:
1. documentation
2. be more consistent with the use of sparse matrices.
3. Potentially remove higher degree activation code
"""

def calc_update(sig, diag_g0_list, diag_g1_list):
    c1c2_mat = np.matmul(
        np.sqrt(diag_g0_list)[:, np.newaxis],
        np.sqrt(diag_g1_list)[np.newaxis, :]
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


def calc_update_diag(sig):
    c1c2_mat = np.matmul(
        np.sqrt(np.diag(sig))[:, np.newaxis],
        np.sqrt(np.diag(sig))[np.newaxis, :]
    )
    lam = sig / c1c2_mat
    lam = np.minimum(np.maximum(lam,-1),1)
    # activation degree = 1
    sig = (lam * (np.pi - np.arccos(lam)) + np.sqrt(1 - lam ** 2)) / (2 * np.pi) * c1c2_mat
    sig = 2 * sig

    return sig


def BLOCK_diag(adj_kprod, sig, scale_mat, R):
    """putting together all block elements (aggregation and R FC layers)"""
    block_diag_list = []
    sig = scale_mat * sparse.csr_matrix.dot(adj_kprod, sig.reshape(-1)).reshape(sig.shape)
    for fc in range(R):
        block_diag_list.append(np.diag(sig))
        sig = calc_update_diag(sig)

    return sig, block_diag_list

def BLOCK(adj_kprod, sig, diag_g0_list, diag_g1_list, theta, scale_mat, R):
    """putting together all block elements (aggregation and R FC layers)"""
    sig = scale_mat * sparse.csr_matrix.dot(adj_kprod, sig.reshape(-1)).reshape(sig.shape)
    theta = scale_mat * sparse.csr_matrix.dot(adj_kprod, theta.reshape(-1)).reshape(theta.shape)

    for fc in range(R):
        sig, sig_dot = calc_update(sig, diag_g0_list[fc], diag_g1_list[fc])
        theta = theta * sig_dot + sig
    return sig, theta


def READOUT(theta, jk=True):
    if not jk:
        return np.array(theta[-1]).sum()
    if jk:
        return np.array(theta).sum()


def gntk_pair(g0_lab, g1_lab, g0_adj, g1_adj, L, R, scale, jk):
    # calculate sig_g0 diag
    if scale == "uniform":
        scale_mat_g0 = np.ones((g0_adj.shape[0], g0_adj.shape[0]))
    elif scale == "degree":
        scale_mat_g0 = 1 / np.matmul(
            np.sum(g0_adj.toarray(), axis=1)[:, np.newaxis],
            np.sum(g0_adj.toarray(), axis=0)[np.newaxis, :]
        )
    adj_kprod_g0 = sparse.kron(g0_adj, g0_adj)
    diag_g0 = []
    sig_g0 = np.matmul(g0_lab, g0_lab.transpose())
    for block in range(L):
        sig_g0, block_diag_list = BLOCK_diag(adj_kprod_g0, sig_g0, scale_mat_g0, R)
        diag_g0.append(block_diag_list)
    del scale_mat_g0, adj_kprod_g0, sig_g0

    # calculate sig_g1 diag
    if scale == "uniform":
        scale_mat_g1 = np.ones((g1_adj.shape[0], g1_adj.shape[0]))
    elif scale == "degree":
        scale_mat_g1 = 1 / np.matmul(
            np.sum(g1_adj.toarray(), axis=1)[:, np.newaxis],
            np.sum(g1_adj.toarray(), axis=0)[np.newaxis, :]
        )
    adj_kprod_g1 = sparse.kron(g1_adj, g1_adj)
    diag_g1 = []
    sig_g1 = np.matmul(g1_lab, g1_lab.transpose())
    for block in range(L):
        sig_g1, block_diag_list = BLOCK_diag(adj_kprod_g1, sig_g1, scale_mat_g1, R)
        diag_g1.append(block_diag_list)
    del scale_mat_g1, adj_kprod_g1, sig_g1

    # calculate theta
    if scale == "uniform":
        scale_mat = np.ones((g0_adj.shape[0], g1_adj.shape[0]))
    elif scale == "degree":
        scale_mat = 1 / np.matmul(
            np.sum(g0_adj.toarray(), axis=1)[:, np.newaxis],
            np.sum(g1_adj.toarray(), axis=0)[np.newaxis, :]
        )
    adj_kprod = sparse.kron(g0_adj, g1_adj)
    sig = np.matmul(g0_lab, g1_lab.transpose())
    theta = sig.copy()
    theta_jk = []
    theta_jk.append(sig.copy())

    for block in range(L):
        sig, theta = BLOCK(adj_kprod, sig, diag_g0[block], diag_g1[block], theta, scale_mat, R)
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
        L, R, scale, jk
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
        L, R, scale, jk
    ) for pair in diag_pair_list)
    gram_items = (gntk_pair(
        lab_list[pair[0]], lab_list[pair[1]],
        adj_list[pair[0]], adj_list[pair[1]],
        L, R, scale, jk
    ) for pair in pair_list)

    for i, item in enumerate(diag_gram_items):
        gram_mat[i,i] = item
    for i, item in enumerate(gram_items):
        gram_mat[pair_list[i][0], pair_list[i][1]] = item
        gram_mat[pair_list[i][1], pair_list[i][0]] = item
        print("\r{}/{}".format(i, len(pair_list)), end='', flush=True)

    return gram_mat