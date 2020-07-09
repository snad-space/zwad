import numpy as np


def load_dataset(oid_file, feature_file, feature_names_file=None):
    """
    Load a dataset.

    Parameters
    ----------
    oid_file: file with stored numpy array of oids
    feature_file: file with stored numpy array of corresponding featues
    feature_names_file: file with features' names (Optional, default: None).
    """
    oid = np.memmap(oid_file, mode='r', dtype=np.uint64)

    if feature_names_file is not None:
        with open(feature_names_file) as fo:
            names = fo.read().split()

        dt = [(name, np.float32) for name in names]
        features = np.memmap(feature_file, mode='r', dtype=dt, shape=oid.shape)
    else:
        features = np.memmap(feature_file, mode='r', dtype=np.float32)
        features = features.reshape(oid.size, -1)

    return oid, features


def scale_values(features, algorithm=None):
    """
    Scale the table of features.

    Parameters
    ----------
    features: Matrix of N-by-M values with N experiments and M features.
    algorithm: Scaling algorithm: 'minmax' or 'std' (default).
    Minmax algorithm scales to the [0; 1] interval, while std algorithm
    scales to the zero mean and unitary standard deviation.

    Return
    ------
    N-by-M matrix of scaled features.
    """
    algorithm = algorithm or 'std'

    if algorithm == 'minmax':
        minis = features.min(axis=0)
        maxis = features.max(axis=0)
        delta = maxis - minis
        delta[delta == 0] = 1.0
        return (features - minis) / delta
    elif algorithm == 'std':
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        return (features - mean) / np.maximum(std, np.finfo(np.float).eps)
