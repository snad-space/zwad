import numpy as np


def scale_values(features, algorithm=None):
    """
    Scale the table of features.

    Parameters
    ----------
    features: Matrix of N-by-M values with N experiments and M features.
    algorithm: Scaling algorithm: 'minmax', 'std' or 'pca' (default).
    Minmax algorithm scales to the [0; 1] interval, while std algorithm
    scales to the zero mean and unitary standard deviation. PCA algorithm
    performs linear transform to principal axes of covariance ellipsoid.

    Return
    ------
    N-by-M matrix of scaled features.
    """
    algorithm = algorithm or 'std'

    if algorithm == 'pca':
        u, _, _ = np.linalg.svd(features, full_matrices=False)
        return u
    elif algorithm == 'minmax':
        minis = features.min(axis=0)
        maxis = features.max(axis=0)
        delta = maxis - minis
        delta[delta == 0] = 1.0
        return (features - minis) / delta
    elif algorithm == 'std':
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        return (features - mean) / np.maximum(std, np.finfo(np.float).eps)
    else:
        raise ValueError('Unkown scale algorithm: {}'.format(algorithm))