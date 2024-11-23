import re
import timeit
import os
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch


def generate_degree_laplacian_matrix(adj_matrix, norm=None):
    g = nx.from_numpy_array(adj_matrix)
    degree = [val for (node, val) in g.degree()]
    if norm == "sym":
        degree_matrix = sp.diags(np.asarray(degree) ** -0.5, dtype=float)
        laplacian_matrix = (
                sp.eye(g.number_of_nodes()) - degree_matrix * adj_matrix * degree_matrix
        )
    elif norm == "walk":
        degree_matrix = sp.diags(np.asarray(degree).clip(1) ** -1.0, dtype=float)
        laplacian_matrix = sp.eye(g.number_of_nodes()) - degree_matrix * adj_matrix
    else:
        degree_matrix = sp.diags(degree, dtype=float)
        laplacian_matrix = degree_matrix * sp.eye(g.number_of_nodes()) - adj_matrix
    return degree_matrix, sp.csr_matrix(laplacian_matrix)


# compute the smallest non-trivial eigenvector of Laplacian
@measure_time
def get_eig(adj_matrix, tol=1e-2, num_eigen=2, norm=None, gamma=0, alpha=0):
    degree_matrix, laplacian_matrix = generate_degree_laplacian_matrix(
        adj_matrix, norm=norm
    )
    if gamma or alpha:
        print("normalizing matrix with alpha and gamma...")
        n_row = laplacian_matrix.shape[0]
        augmented_matrix = gamma * degree_matrix + (1 - gamma) * sp.eye(n_row)
        # normalizer = sp.diags((normalizer.diagonal()) ** -1.0, dtype=float)
        # laplacian_matrix = normalizer * laplacian_matrix
        l_normalizer = sp.diags((augmented_matrix.diagonal()) ** -alpha, dtype=float)
        r_normalizer = sp.diags((augmented_matrix.diagonal()) ** (1 - alpha), dtype=float)
        laplacian_matrix = gamma * l_normalizer * laplacian_matrix * r_normalizer

    eig_vals, eig_vecs = sp.linalg.eigs(
        laplacian_matrix, k=num_eigen, which="SR", tol=tol
    )
    print("The second smallest eigenvalue of Ls via library: %.4f" % (eig_vals[-1]))
    eigen_vec = torch.from_numpy(np.real(eig_vecs[:, 1])).float()
    return torch.reshape(eigen_vec, (-1,))

