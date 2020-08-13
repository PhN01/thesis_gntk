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
1. Cleanup
2. Documentation
    
Identified issues:
2. be more consistent with the use of sparse matrices
"""

def calc_update(
        sig,
        diag_g0_list,
        diag_g1_list
):
    """
    routine to update the sigma matrix for g0 != g1
    :param sig: previous sigma matrix of g0 with g1
    :param diag_g0_list: array list containing the diagonal values of sigma_g0 at each
    step of the graph neural network. structure of the array list is
    (num_blocks, num_fc_layers, n_nodes_g0)
    :param diag_g1_list: array list containing the diagonal values of sigma_g1 at each
    step of the graph neural network. structure of the array list is
    (num_blocks, num_fc_layers, n_nodes_g1)
    :return: returns a tuple of the updated sigma and sigma_dot matrices
    """

    # calculate a matrix of c1*c2 values as in the definition of the
    # closed form updates of sigma. shape: (n_nodes_g0 x n_nodes_g1)
    c1c2_mat = np.matmul(
        np.sqrt(diag_g0_list)[:, np.newaxis],
        np.sqrt(diag_g1_list)[np.newaxis, :]
    )

    # calculate matrix of lambda values as in the definition of the
    # closed form updates of sigma
    lam = sig / c1c2_mat
    # clip lambda values at {-1,1}. otherwise the next step is not well
    # defined
    lam = np.minimum(np.maximum(lam,-1),1)

    # calculate update of composition arc-cosine kernel of degree 1
    # according to cho & saul (2009)
    sig = (lam * (np.pi - np.arccos(lam)) + np.sqrt(1 - lam ** 2)) / (2 * np.pi) * c1c2_mat
    # scaling factor csig = 2
    sig = 2 * sig

    # calculate update of composition arc-cosine kernel of degree 0
    # according to cho & saul (2009)
    sig_dot = (np.pi - np.arccos(lam)) / (2 * np.pi)
    # scaling factor csig = 2
    sig_dot = 2 * sig_dot

    return sig, sig_dot


def calc_update_diag(
        sig
):
    """
    routine to update the sigma matrix for g0 == g1. sigma_dot is not
    required for this case
    :param sig: previous sigma matrix of g0 with g0
    :return: returns the updated sigma matrix
    """

    # calculate a matrix of c1*c2 values as in the definition of the
    # closed form updates of sigma. shape: (n_nodes_g0 x n_nodes_g1)
    c1c2_mat = np.matmul(
        np.sqrt(np.diag(sig))[:, np.newaxis],
        np.sqrt(np.diag(sig))[np.newaxis, :]
    )

    # calculate matrix of lambda values as in the definition of the
    # closed form updates of sigma
    lam = sig / c1c2_mat
    # clip lambda values at {-1,1}. otherwise the next step is not well
    # defined
    lam = np.minimum(np.maximum(lam,-1),1)

    # calculate update of composition arc-cosine kernel of degree 1
    # according to cho & saul (2009)
    sig = (lam * (np.pi - np.arccos(lam)) + np.sqrt(1 - lam ** 2)) / (2 * np.pi) * c1c2_mat
    # scaling factor csig = 2
    sig = 2 * sig

    return sig


def BLOCK_diag(
        adj_kprod,
        sig,
        scale_mat,
        R
):
    """
    perform the updates of an entire block layer to sigma for g0 == g1 according to du et al.
    :param adj_kprod: Sparse Kronecker product of the adjacency matrices of the graphs.
    Since g0 == g1, we have dimensionality (n_nodes_g0^2 x n_nodes_g0^2)
    :param sig: previous sigma matrix of g0 with g0 from previous block layer
    :param scale_mat: Scaling matrix of the graphs. Since g0 == g1, this is in
    (n_nodes_g0 x n_nodes_g0)
    :param R: GNTK parameter for the number of fc layers per block layer
    :return: returns a tuple of the updated sigma matrix after the block and a list of lists
    of the diagonal values of sigma at fc layer
    """

    # initialize empty list that will contain lists of the diagonal values of the
    # sigma values at each fc layer
    block_diag_list = []

    # aggregate the sigma values of each node and their respective neighboring
    # nodes. scale the result by the scale_mat (set according to GNTK parameter
    # scale).
    sig = scale_mat * sparse.csr_matrix.dot(adj_kprod, sig.reshape(-1)).reshape(sig.shape)

    # iterate over the fc layers of the block layer
    for fc in range(R):

        # append the diagonal values of the previous sigma matrix to block_diag_list
        # the idea is that in the calculation of the sigma for g0 != g1, the diagonal
        # values of sigma_g0 and sigma_g1 at the previous step are required
        block_diag_list.append(np.diag(sig))

        # update the sigma matrix
        sig = calc_update_diag(sig)

    return sig, block_diag_list

def BLOCK(
        adj_kprod,
        sig,
        diag_g0_list,
        diag_g1_list,
        theta,
        scale_mat,
        R
):
    """
    perform the updates of an entire block layer to sigma for g0 != g1 according to du et al.
    :param adj_kprod: Sparse Kronecker product of the adjacency matrices of the graphs.
    Since g0 != g1, we have dimensionality (n_nodes_g0^2 x n_nodes_g1^2)
    :param sig: previous sigma matrix of g0 with g1 from previous block layer
    :param diag_g0_list: array list of lists of the diagonal values of of sigma_g0 at the current
    block. The dimensionality of the array list is (num_fc_layers, n_nodes_g0)
    :param diag_g1_list: array list of lists of the diagonal values of of sigma_g1 at the current
    block. The dimensionality of the array list is (num_fc_layers, n_nodes_g1)
    :param theta: previous theta matrix of g0 with g1 from previous block layer
    :param scale_mat: Scaling matrix of the graphs. Since g0 != g1, this is in
    (n_nodes_g0 x n_nodes_g1)
    :param R: GNTK parameter for the number of fc layers per block layer
    :return: returns a tuple of the updated sigma and theta matrices
    """

    # aggregate the sigma and theta values of each node and their respective neighboring
    # nodes. scale the result by the scale_mat (set according to GNTK parameter
    # scale).
    sig = scale_mat * sparse.csr_matrix.dot(adj_kprod, sig.reshape(-1)).reshape(sig.shape)
    theta = scale_mat * sparse.csr_matrix.dot(adj_kprod, theta.reshape(-1)).reshape(theta.shape)

    # iterate over the fc layers of the block layer
    for fc in range(R):

        # update the sigma matrix
        sig, sig_dot = calc_update(sig, diag_g0_list[fc], diag_g1_list[fc])

        # update the theta matrix
        theta = theta * sig_dot + sig

    return sig, theta


def READOUT(
        theta,
        jk=True
):
    """
    function to perform the readout operation at the end of the gntk calculations
    according to du et al. (2019) to get a final theta value
    :param theta: list containing the theta matrices after each block layer, hence
    len(theta) == num_block_layers == L
    :param jk: GNTK parameter indicating if jumping knowledge is used in the graph
    neural network
    :return: returns a scalar kernel value for the similarity between g0 and g1
    """

    # case switch: if jumping knowledge is activated or not
    if not jk:
        # if no jumping knowledge is present, only sum over the theta values of the last
        # block layer
        return np.array(theta[-1]).sum()
    if jk:
        # if jumping knowledge is present, sum over the theta values of all block layers
        return np.array(theta).sum()


def gntk_pair(
        g0_lab,
        g1_lab,
        g0_adj,
        g1_adj,
        L,
        R,
        scale,
        jk
):
    """
    main gntk routine to calculate the pair wise kernel value between to graphs g0 and g1
    :param g0_lab: label matrix of graph g0 with dimensionality
    (n_nodes_g0 x dimensionality_onehot_labels)
    :param g1_lab: label matrix of graph g1 with dimensionality
    (n_nodes_g1 x dimensionality_onehot_labels)
    :param g0_adj: adjacency matrix of graph g0 with self-connections (diag(g0_adj) = 1).
    dimensionality of the adjacency matrix is (n_nodes_g0 x n_nodes_g0)
    :param g1_adj: adjacency matrix of graph g1 with self-connections (diag(g1_adj) = 1).
    dimensionality of the adjacency matrix is (n_nodes_g1 x n_nodes_g1)
    :param L: GNTK parameter for the number of block layers of the graph neural net
    :param R: GNTK parameter for the number of fc layers per block layer
    :param scale: GNTK parameter indicating how to scale the aggregation result. Possible
    values are 'uniform' (no scaling) and 'degree' (scaling by the degree of the nodes)
    :param jk: GNTK parameter indicating if jumping knowledge is used in the graph
    neural network
    :return: returns a scalar kernel value for the similarity between g0 and g1
    """

    # initialize a dict to store the array lists for the diagonal values of the
    # sigma matrices of g0 and g1 with themselves
    diags = {
        'diag_g0': None,
        'diag_g1': None
    }

    # loop over both graphs to calculate the sigma diagonals for the respective
    # graphs
    for i, (graph_lab, graph_adj) in \
        enumerate(zip([g0_lab, g1_lab], [g0_adj, g1_adj])):

        # case switch: prepare scaling matrix according to GNTK parameter {scale}
        # dimensionality of scale_mat_graph is (n_nodes_graph x n_nodes_graph)
        if scale == "uniform":
            scale_mat_graph = np.ones(
                (graph_adj.shape[0], graph_adj.shape[0])
            )
        elif scale == "degree":
            scale_mat_graph = 1 / np.matmul(
                np.sum(graph_adj.toarray(), axis=1)[:, np.newaxis],
                np.sum(graph_adj.toarray(), axis=0)[np.newaxis, :]
            )

        # sparse calculation of the Kronecker product of the the adjacency matrix of
        # the graph. dimensionality: (n_nodes_graph^2 x n_nodes_graph^2)
        adj_kprod_graph = sparse.kron(graph_adj, graph_adj)

        # initialize diag array list for current graph
        diag_graph = []

        # calculate inital sigma matrix for the graph. dimensionality: (n_nodes_graph x
        # n_nodes_graph)
        sig_graph = np.matmul(graph_lab, graph_lab.transpose())

        # iterate over the block layers
        for block in range(L):

            # get the updated sigma matrix after each block as well as the list
            # of lists with diagonal values of sigma after each fc layer within
            # the block
            sig_graph, block_diag_list = BLOCK_diag(adj_kprod_graph, sig_graph, scale_mat_graph, R)
            diag_graph.append(block_diag_list)

        # delete matrices that are not needed anymore
        del scale_mat_graph, adj_kprod_graph, sig_graph

        # write completed array list of diagonal values to diags
        diags[f'diag_g{i}'] = diag_graph

    diag_g0 = diags['diag_g0']
    diag_g1 = diags['diag_g1']

    # case switch: prepare scaling matrix according to GNTK parameter {scale}
    # dimensionality of scale_mat is (n_nodes_g0 x n_nodes_g1)
    if scale == "uniform":
        scale_mat = np.ones((g0_adj.shape[0], g1_adj.shape[0]))
    elif scale == "degree":
        scale_mat = 1 / np.matmul(
            np.sum(g0_adj.toarray(), axis=1)[:, np.newaxis],
            np.sum(g1_adj.toarray(), axis=0)[np.newaxis, :]
        )

    # sparse calculation of the Kronecker product of the the adjacency matrices of
    # g0 and g1. dimensionality: (n_nodes_g0^2 x n_nodes_g1^2)
    adj_kprod = sparse.kron(g0_adj, g1_adj)

    # calculate inital sigma matrix for the graph. dimensionality: (n_nodes_g0 x
    # n_nodes_g1)
    sig = np.matmul(g0_lab, g1_lab.transpose())
    # initalize theta (theta_0 == sigma_0)
    theta = sig.copy()

    # initialize a list for appending theta after each block
    theta_jk = []
    theta_jk.append(sig.copy())

    # iterate over the block layers
    for block in range(L):

        # get the updated sigma and theta matrices after each block
        sig, theta = BLOCK(adj_kprod, sig, diag_g0[block], diag_g1[block], theta, scale_mat, R)
        theta_jk.append(theta.copy())

    # calculate the final kernel value depending on the GNTK parameter {jk}
    theta = READOUT(theta_jk, jk)

    return theta


def gntk_gram_joblib(
        lab_list,
        adj_list,
        L,
        R,
        scale,
        jk,
        n_jobs
):
    """
    routine to calculate the gram matrix for an entire graph dataset parallelized
    :param lab_list: list of label matrices of each graph in the dataset.
    len(lab_list) == n_graphs
    :param adj_list: list of adjacency matrices of each graph in the dataset.
    len(adj_list) == n_graphs. adjacency matrices must contain self connections.
    :param L: GNTK parameter for the number of block layers of the graph neural net
    :param R: GNTK parameter for the number of fc layers per block layer
    :param scale: GNTK parameter indicating how to scale the aggregation result. Possible
    values are 'uniform' (no scaling) and 'degree' (scaling by the degree of the nodes)
    :param jk: GNTK parameter indicating if jumping knowledge is used in the graph
    neural network
    :param n_jobs: number of jobs to parallelize the calculation of gram kernel values
    over
    :return: returns the final gram matrix for the graph dataset
    """

    n_graphs = len(lab_list)

    # create a list of all possible pairs of indices for the graphs
    pair_list = [(i, j) for i in range(n_graphs) for j in range(i, n_graphs)]

    # Parallelize the calculation of kernel values for each graph pair
    # returns a list of kernel values in the same order as pair_list
    results = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(gntk_pair)(
        lab_list[pair[0]],lab_list[pair[1]],
        adj_list[pair[0]],adj_list[pair[1]],
        L, R, scale, jk
    ) for pair in tqdm(pair_list))

    # initialize the gram matrix
    gram_mat = np.zeros((n_graphs, n_graphs))

    # iterate over all pairwise kernel values and write them into the
    # gram matrix according to the indices in pair_list
    for i, val in enumerate(results):
        pair = pair_list[i]
        gram_mat[pair[0], pair[1]] = val
        gram_mat[pair[1], pair[0]] = val

    return gram_mat


def gntk_gram_profiling(
        lab_list,
        adj_list,
        L,
        R,
        scale,
        jk
):
    """
    Non-parallelized gram matrix calculation routine that should not be used normally.
    We use this to analyze the computational complexity of the steps of above methods.
    Therefore each step is coded explicitly in this function.
    :param lab_list: list of label matrices of each graph in the dataset.
    len(lab_list) == n_graphs
    :param adj_list: list of adjacency matrices of each graph in the dataset.
    len(adj_list) == n_graphs. adjacency matrices must contain self connections.
    :param L: GNTK parameter for the number of block layers of the graph neural net
    :param R: GNTK parameter for the number of fc layers per block layer
    :param scale: GNTK parameter indicating how to scale the aggregation result. Possible
    values are 'uniform' (no scaling) and 'degree' (scaling by the degree of the nodes)
    :param jk: GNTK parameter indicating if jumping knowledge is used in the graph
    neural network
    :return: returns the final gram matrix for the graph dataset
    """
    # preparation
    n_graphs = len(lab_list)
    pair_list = [(i, j) for i in range(n_graphs) for j in range(i, n_graphs)]
    n_pairs = len(pair_list)
    gram_mat = np.zeros((n_graphs, n_graphs))

    theta_list = []

    # calculate GNTK kernel value for each pair of graphs
    for i, pair in enumerate(pair_list):

        # case switch: prepare scaling matrices according to GNTK parameter {scale}
        if scale == "uniform":
            cu_mat = np.ones((adj_list[pair[0]].shape[0], adj_list[pair[1]].shape[0]))
            cu_mat_g0 = np.ones((adj_list[pair[0]].shape[0], adj_list[pair[0]].shape[0]))
            cu_mat_g1 = np.ones((adj_list[pair[1]].shape[0], adj_list[pair[1]].shape[0]))
        elif scale == "degree":
            cu_mat = 1 / np.matmul(np.sum(adj_list[pair[0]].toarray(), axis=1)[:, np.newaxis],
                                   np.sum(adj_list[pair[1]].toarray(), axis=0)[np.newaxis, :])
            cu_mat_g0 = 1 / np.matmul(np.sum(adj_list[pair[0]].toarray(), axis=1)[:, np.newaxis],
                                      np.sum(adj_list[pair[0]].toarray(), axis=0)[np.newaxis, :])
            cu_mat_g1 = 1 / np.matmul(np.sum(adj_list[pair[1]].toarray(), axis=1)[:, np.newaxis],
                                      np.sum(adj_list[pair[1]].toarray(), axis=0)[np.newaxis, :])

        # kronecker product matrix for the aggregation step
        adjk = sparse.kron(adj_list[pair[0]], adj_list[pair[1]])
        adjk_g0 = sparse.kron(adj_list[pair[0]], adj_list[pair[0]])
        adjk_g1 = sparse.kron(adj_list[pair[1]], adj_list[pair[1]])

        # initialization of sigma matrices and theta matrix
        sig = np.matmul(lab_list[pair[0]], lab_list[pair[1]].transpose())
        sig_g0 = np.matmul(lab_list[pair[0]], lab_list[pair[0]].transpose())
        sig_g1 = np.matmul(lab_list[pair[1]], lab_list[pair[1]].transpose())
        theta = sig.copy()

        # theta list to store theta after each block
        theta_jk = []
        theta_jk.append(sig.copy())

        # process the sigma and theta matrices through L block layers
        for block in range(1, L + 1):

            # aggregate all matrices
            sig = cu_mat * sparse.csr_matrix.dot(adjk, sig.reshape(-1)).reshape(sig.shape)
            sig_g0 = cu_mat_g0 * sparse.csr_matrix.dot(adjk_g0, sig_g0.reshape(-1)).reshape(sig_g0.shape)
            sig_g1 = cu_mat_g1 * sparse.csr_matrix.dot(adjk_g1, sig_g1.reshape(-1)).reshape(sig_g1.shape)
            theta = cu_mat * sparse.csr_matrix.dot(adjk, theta.reshape(-1)).reshape(theta.shape)

            # propagate sigma matrices through R fully connected layers
            for fc in range(R):
                # series of calculations for preparing the sigma updates
                c1c2_mat = np.matmul(
                    np.sqrt(np.diag(sig_g0))[:, np.newaxis],
                    np.sqrt(np.diag(sig_g1))[np.newaxis, :]
                )
                lam = sig / c1c2_mat
                lam = np.minimum(np.maximum(lam, -1), 1)

                # sigma update using non-updated sig_g0 and sig_g1
                sig = (lam * (np.pi - np.arccos(lam)) + np.sqrt(1 - lam ** 2)) / (2 * np.pi) * c1c2_mat
                sig = 2 * sig
                sig_dot = (np.pi - np.arccos(lam)) / (2 * np.pi)
                sig_dot = 2 * sig_dot

                # theta update
                theta = theta * sig_dot + sig

                # series of calculations for preparing sig_g0 update
                c1c2_mat = np.matmul(
                    np.sqrt(np.diag(sig_g0))[:, np.newaxis],
                    np.sqrt(np.diag(sig_g0))[np.newaxis, :]
                )
                lam = sig_g0 / c1c2_mat
                lam = np.minimum(np.maximum(lam, -1), 1)

                # sig_g0 update
                sig_g0 = (lam * (np.pi - np.arccos(lam)) + np.sqrt(1 - lam ** 2)) / (2 * np.pi) * c1c2_mat
                sig_g0 = 2 * sig_g0

                # series of calculations for preparing sig_g1 update
                c1c2_mat = np.matmul(
                    np.sqrt(np.diag(sig_g1))[:, np.newaxis],
                    np.sqrt(np.diag(sig_g1))[np.newaxis, :]
                )
                lam = sig_g1 / c1c2_mat
                lam = np.minimum(np.maximum(lam, -1), 1)

                # sig_g1 update
                sig_g1 = (lam * (np.pi - np.arccos(lam)) + np.sqrt(1 - lam ** 2)) / (2 * np.pi) * c1c2_mat
                sig_g1 = 2 * sig_g1

            # store theta after propagation through full block layer
            theta_jk.append(theta.copy())

        # store the final GNTK kernel value for the given pair
        theta_list.append(READOUT(theta_jk, jk))

    # populate the gram matrix with the GNTK kernel values
    for i, pair in enumerate(pair_list):
        gram_mat[pair[0], pair[1]] = theta_list[i]
        gram_mat[pair[1], pair[0]] = theta_list[i]

    return gram_mat