# from sklearn.utils.graph import graph_laplacian
# from sklearn.utils.arpack import eigsh
# from sklearn.manifold.spectral_embedding_ import _set_diag
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy import sparse
from scipy.sparse.linalg import eigsh

import numpy as np


def _set_diag(laplacian, value, norm_laplacian=True):
    n_nodes = laplacian.shape[0]
    if not sparse.isspmatrix(laplacian):
        if norm_laplacian:
            laplacian.flat[:: n_nodes + 1] = value
    else:
        laplacian = laplacian.tocoo()
        if norm_laplacian:
            diag_idx = laplacian.row == laplacian.col
            laplacian.data[diag_idx] = value
        n_diags = np.unique(laplacian.row - laplacian.col).size
        if n_diags <= 7:
            laplacian = laplacian.todia()
        else:
            laplacian = laplacian.tocsr()
    return laplacian


def predict_k(affinity_matrix):
    # normed_laplacian, dd = graph_laplacian(affinity_matrix, normed=True, return_diag=True)
    normed_laplacian, dd = csgraph_laplacian(affinity_matrix, normed=True, return_diag=True)
    # laplacian = _set_diag(normed_laplacian, 1)
    laplacian = _set_diag(normed_laplacian, 1)

    n_components = affinity_matrix.shape[0] - 1

    # eigenvalues, eigenvectors = eigsh(-laplacian, k=n_components, which="LM", sigma=1.0, maxiter=5000)
    eigenvalues, eigenvectors = eigsh(
        -laplacian, k=n_components, sigma=1.0, which="LM", maxiter=5000
    )
    eigenvalues = -eigenvalues[::-1]

    max_gap = 0
    gap_pre_index = 0
    for i in range(1, eigenvalues.size):
        gap = eigenvalues[i] - eigenvalues[i - 1]
        if gap > max_gap:
            max_gap = gap
            gap_pre_index = i - 1

    k = gap_pre_index + 1

    return k


if __name__ == "__main__":
    a = np.array([[0, 1, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 1, 0]])
    k = predict_k(a)
    print(k)
