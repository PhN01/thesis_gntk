"""
@author: Philipp Nikolaus
@date: 25.03.2020
@references: Du, Simon et al. (2019): Graph Neural Tangent Kernel - Fusing Graph Neural Networks and Graph Kernels
"""

import numpy as np
import scipy.sparse as sparse
import joblib
from tqdm import tqdm


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
    Args:
        lab_list (): list of label matrices of each graph in the dataset.
            len(lab_list) == n_graphs
        adj_list (): list of adjacency matrices of each graph in the dataset.
            len(adj_list) == n_graphs. adjacency matrices must contain self connections.
        L (): GNTK parameter for the number of block layers of the graph neural net
        R (): GNTK parameter for the number of fc layers per block layer
        scale (): GNTK parameter indicating how to scale the aggregation result. Possible
            values are 'uniform' (no scaling) and 'degree' (scaling by the degree of the nodes)
        jk (): GNTK parameter indicating if jumping knowledge is used in the graph
            neural network
    Returns:
        returns the final gram matrix for the graph dataset
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
        theta_list.append(_READOUT(theta_jk, jk))

    # populate the gram matrix with the GNTK kernel values
    for i, pair in enumerate(pair_list):
        gram_mat[pair[0], pair[1]] = theta_list[i]
        gram_mat[pair[1], pair[0]] = theta_list[i]

    return gram_mat


def _READOUT(theta, jk):
        """
        function to perform the readout operation at the end of the gntk calculations
        according to du et al. (2019) to get a final theta value
        Args:
            theta (): list containing the theta matrices after each block layer, hence
                len(theta) == num_block_layers == L
            jk (): GNTK parameter indicating if jumping knowledge is used in the graph
                neural network
        Returns:
            returns a scalar kernel value for the similarity between g0 and g1
        """

        # case switch: if jumping knowledge is activated or not
        if not jk:
            # if no jumping knowledge is present, only sum over the theta values of the last
            # block layer
            return np.array(theta[-1]).sum()
        if jk:
            # if jumping knowledge is present, sum over the theta values of all block layers
            return np.array(theta).sum()