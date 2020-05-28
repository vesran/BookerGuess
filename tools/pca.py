import numpy as np


def PCA(data, k=2):
    """ PCA
    :param data: numpy array
    :param k: final dimensionality
    :return: data with k dimensionality
    """
    data = data - data.mean(axis=0)
    data = data / data.std(axis=0)
    R = np.cov(data.T, rowvar=True)
    evals, evecs = np.linalg.eig(R)

    # Sort eigenvalue in decreasing order and select the k corresponding
    # largest eigenvectors
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evecs = evecs[:, :k]
    return np.dot(data, evecs)
