"""
@author: Philipp Nikolaus
@date: 25.03.2020
@references: Du, Simon et al. (2019): Graph Neural Tangent Kernel - Fusing Graph Neural Networks and Graph Kernels
"""

import numpy as np
import scipy.sparse as sparse
import joblib
from tqdm import tqdm


class GNTK:
    def __init__(self, L, R, scale, jk):
        self.L = L
        self.R = R
        self.scale = scale
        self.jk = jk

    def gntk_gram(self, lab_list, adj_list, n_jobs=-1, verbose=1):
        """
        routine to calculate the gram matrix for an entire graph dataset parallelized
        Args:
            lab_list (): list of label matrices of each graph in the dataset.
                len(lab_list) == n_graphs
            adj_list (): list of adjacency matrices of each graph in the dataset.
                len(adj_list) == n_graphs. adjacency matrices must contain self connections.
            n_jobs (): number of jobs to parallelize the calculation of gram kernel values
            over
        Returns:
            returns the final gram matrix for the graph dataset
        """

        n_graphs = len(lab_list)

        # create a list of all possible pairs of indices for the graphs
        pair_list = [(i, j) for i in range(n_graphs) for j in range(i, n_graphs)]

        # Parallelize the calculation of kernel values for each graph pair
        # returns a list of kernel values in the same order as pair_list
        if verbose:
            results = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(self.gntk_pair)(
                    lab_list[pair[0]],
                    lab_list[pair[1]],
                    adj_list[pair[0]],
                    adj_list[pair[1]],
                )
                for pair in tqdm(pair_list)
            )
        else:
            results = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(self.gntk_pair)(
                    lab_list[pair[0]],
                    lab_list[pair[1]],
                    adj_list[pair[0]],
                    adj_list[pair[1]],
                )
                for pair in pair_list
            )

        # initialize the gram matrix
        gram_mat = np.zeros((n_graphs, n_graphs))

        # iterate over all pairwise kernel values and write them into the
        # gram matrix according to the indices in pair_list
        for i, val in enumerate(results):
            pair = pair_list[i]
            gram_mat[pair[0], pair[1]] = val
            gram_mat[pair[1], pair[0]] = val

        return gram_mat

    def gntk_pair(self, g0_lab, g1_lab, g0_adj, g1_adj):
        """
        main gntk routine to calculate the pair wise kernel value between to graphs g0 and g1
        Args:
            g0_lab (): label matrix of graph g0 with dimensionality
                (n_nodes_g0 x dimensionality_onehot_labels)
            g1_lab ():  label matrix of graph g1 with dimensionality
                (n_nodes_g1 x dimensionality_onehot_labels)
            g0_adj (): adjacency matrix of graph g0 with self-connections (diag(g0_adj) = 1).
                dimensionality of the adjacency matrix is (n_nodes_g0 x n_nodes_g0)
            g1_adj (): adjacency matrix of graph g1 with self-connections (diag(g1_adj) = 1).
                dimensionality of the adjacency matrix is (n_nodes_g1 x n_nodes_g1)
        Returns:
            returns a scalar kernel value for the similarity between g0 and g1
        """

        # initialize a dict to store the array lists for the diagonal values of the
        # sigma matrices of g0 and g1 with themselves
        diags = {"diag_g0": None, "diag_g1": None}

        # loop over both graphs to calculate the sigma diagonals for the respective
        # graphs
        for i, (graph_lab, graph_adj) in enumerate(
            zip([g0_lab, g1_lab], [g0_adj, g1_adj])
        ):

            # case switch: prepare scaling matrix according to GNTK parameter {scale}
            # dimensionality of scale_mat_graph is (n_nodes_graph x n_nodes_graph)
            if self.scale == "uniform":
                scale_mat_graph = np.ones((graph_adj.shape[0], graph_adj.shape[0]))
            elif self.scale == "degree":
                scale_mat_graph = 1 / np.matmul(
                    np.sum(graph_adj.toarray(), axis=1)[:, np.newaxis],
                    np.sum(graph_adj.toarray(), axis=0)[np.newaxis, :],
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
            for block in range(self.L):

                # get the updated sigma matrix after each block as well as the list
                # of lists with diagonal values of sigma after each fc layer within
                # the block
                sig_graph, block_diag_list = self._BLOCK_diag(
                    adj_kprod_graph, sig_graph, scale_mat_graph
                )
                diag_graph.append(block_diag_list)

            # delete matrices that are not needed anymore
            del scale_mat_graph, adj_kprod_graph, sig_graph

            # write completed array list of diagonal values to diags
            diags[f"diag_g{i}"] = diag_graph

        diag_g0 = diags["diag_g0"]
        diag_g1 = diags["diag_g1"]

        # case switch: prepare scaling matrix according to GNTK parameter {scale}
        # dimensionality of scale_mat is (n_nodes_g0 x n_nodes_g1)
        if self.scale == "uniform":
            scale_mat = np.ones((g0_adj.shape[0], g1_adj.shape[0]))
        elif self.scale == "degree":
            scale_mat = 1 / np.matmul(
                np.sum(g0_adj.toarray(), axis=1)[:, np.newaxis],
                np.sum(g1_adj.toarray(), axis=0)[np.newaxis, :],
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
        for block in range(self.L):

            # get the updated sigma and theta matrices after each block
            sig, theta = self._BLOCK(
                adj_kprod, sig, diag_g0[block], diag_g1[block], theta, scale_mat
            )
            theta_jk.append(theta.copy())

        # calculate the final kernel value depending on the GNTK parameter {jk}
        theta = self._READOUT(theta_jk)

        return theta

    def _calc_update(self, sig, diag_g0_list, diag_g1_list):
        """
        Routine to update the sigma matrix for g0 != g1
        Args:
            sig (): previous sigma matrix of g0 with g1.
            diag_g0_list (): array list containing the diagonal values of sigma_g0 at each
                step of the graph neural network. structure of the array list is
                (num_blocks, num_fc_layers, n_nodes_g0).
            diag_g1_list (): array list containing the diagonal values of sigma_g1 at each
                step of the graph neural network. structure of the array list is
                (num_blocks, num_fc_layers, n_nodes_g1)
        Returns:
            returns a tuple of the updated sigma and sigma_dot matrices
        """
        # calculate a matrix of c1*c2 values as in the definition of the
        # closed form updates of sigma. shape: (n_nodes_g0 x n_nodes_g1)
        c1c2_mat = np.matmul(
            np.sqrt(diag_g0_list)[:, np.newaxis], np.sqrt(diag_g1_list)[np.newaxis, :]
        )

        # calculate matrix of lambda values as in the definition of the
        # closed form updates of sigma
        lam = sig / c1c2_mat
        # clip lambda values at {-1,1}. otherwise the next step is not well
        # defined
        lam = np.minimum(np.maximum(lam, -1), 1)

        # calculate update of composition arc-cosine kernel of degree 1
        # according to cho & saul (2009)
        sig = (
            (lam * (np.pi - np.arccos(lam)) + np.sqrt(1 - lam ** 2))
            / (2 * np.pi)
            * c1c2_mat
        )
        # scaling factor csig = 2
        sig = 2 * sig

        # calculate update of composition arc-cosine kernel of degree 0
        # according to cho & saul (2009)
        sig_dot = (np.pi - np.arccos(lam)) / (2 * np.pi)
        # scaling factor csig = 2
        sig_dot = 2 * sig_dot

        return sig, sig_dot

    def _calc_update_diag(self, sig):
        """
        Routine to update the sigma matrix for g0 == g1. sigma_dot is not .
        Args:
            sig (): previous sigma matrix of g0 with g0.
        Returns:
            returns the updated sigma matrix
        """

        # calculate a matrix of c1*c2 values as in the definition of the
        # closed form updates of sigma. shape: (n_nodes_g0 x n_nodes_g1)
        c1c2_mat = np.matmul(
            np.sqrt(np.diag(sig))[:, np.newaxis], np.sqrt(np.diag(sig))[np.newaxis, :]
        )

        # calculate matrix of lambda values as in the definition of the
        # closed form updates of sigma
        lam = sig / c1c2_mat
        # clip lambda values at {-1,1}. otherwise the next step is not well
        # defined
        lam = np.minimum(np.maximum(lam, -1), 1)

        # calculate update of composition arc-cosine kernel of degree 1
        # according to cho & saul (2009)
        sig = (
            (lam * (np.pi - np.arccos(lam)) + np.sqrt(1 - lam ** 2))
            / (2 * np.pi)
            * c1c2_mat
        )
        # scaling factor csig = 2
        sig = 2 * sig

        return sig

    def _BLOCK_diag(self, adj_kprod, sig, scale_mat):
        """
        Perform the updates of an entire block layer to sigma for g0 == g1 according to du et al.
        Args:
            adj_kprod (): Sparse Kronecker product of the adjacency matrices of the graphs.
                Since g0 == g1, we have dimensionality (n_nodes_g0^2 x n_nodes_g0^2).
            sig (): previous sigma matrix of g0 with g0 from previous block layer
            scale_mat (): Scaling matrix of the graphs. Since g0 == g1, this is in
                (n_nodes_g0 x n_nodes_g0)
        Returns:
            returns a tuple of the updated sigma matrix after the block and a list of lists
            of the diagonal values of sigma at fc layer
        """

        # initialize empty list that will contain lists of the diagonal values of the
        # sigma values at each fc layer
        block_diag_list = []

        # aggregate the sigma values of each node and their respective neighboring
        # nodes. scale the result by the scale_mat (set according to GNTK parameter
        # scale).
        sig = scale_mat * sparse.bsr_matrix.dot(adj_kprod, sig.reshape(-1)).reshape(
            sig.shape
        )

        # iterate over the fc layers of the block layer
        for fc in range(self.R):

            # append the diagonal values of the previous sigma matrix to block_diag_list
            # the idea is that in the calculation of the sigma for g0 != g1, the diagonal
            # values of sigma_g0 and sigma_g1 at the previous step are required
            block_diag_list.append(np.diag(sig))

            # update the sigma matrix
            sig = self._calc_update_diag(sig)

        return sig, block_diag_list

    def _BLOCK(self, adj_kprod, sig, diag_g0_list, diag_g1_list, theta, scale_mat):
        """
        perform the updates of an entire block layer to sigma for g0 != g1 according to du et al.
        Args:
            adj_kprod (): Sparse Kronecker product of the adjacency matrices of the graphs.
                Since g0 != g1, we have dimensionality (n_nodes_g0^2 x n_nodes_g1^2)
            sig (): previous sigma matrix of g0 with g1 from previous block layer
            diag_g0_list (): array list of lists of the diagonal values of of sigma_g0 at the current
                block. The dimensionality of the array list is (num_fc_layers, n_nodes_g0)
            diag_g1_list (): array list of lists of the diagonal values of of sigma_g1 at the current
                block. The dimensionality of the array list is (num_fc_layers, n_nodes_g1)
            theta (): previous theta matrix of g0 with g1 from previous block layer
            scale_mat (): Scaling matrix of the graphs. Since g0 != g1, this is in
                (n_nodes_g0 x n_nodes_g1)
        Returns:
            returns a tuple of the updated sigma and theta matrices
        """

        # aggregate the sigma and theta values of each node and their respective neighboring
        # nodes. scale the result by the scale_mat (set according to GNTK parameter
        # scale).
        sig = scale_mat * sparse.bsr_matrix.dot(adj_kprod, sig.reshape(-1)).reshape(
            sig.shape
        )
        theta = scale_mat * sparse.bsr_matrix.dot(adj_kprod, theta.reshape(-1)).reshape(
            theta.shape
        )

        # iterate over the fc layers of the block layer
        for fc in range(self.R):

            # update the sigma matrix
            sig, sig_dot = self._calc_update(sig, diag_g0_list[fc], diag_g1_list[fc])

            # update the theta matrix
            theta = theta * sig_dot + sig

        return sig, theta

    def _READOUT(self, theta):
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
        if not self.jk:
            # if no jumping knowledge is present, only sum over the theta values of the last
            # block layer
            return np.array(theta[-1]).sum()
        if self.jk:
            # if jumping knowledge is present, sum over the theta values of all block layers
            return np.array(theta).sum()

