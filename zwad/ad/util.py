import numpy as np
from tqdm import tnrange

from sklearn.neighbors import LocalOutlierFactor


def run_classifier(classifier, values, jobs=None):
    """Helper function for running sklearn models for AD.

    Parameters
    ----------
    classifier: Instantiated model to run.
    values: numpy array of vectored objects for classification.
    jobs: number of jobs for fitting parallelization.

    Returns
    -------
    Tuple of indices and corresponding scores from the most anomalous to the least.
    """
    if jobs:
        classifier.n_jobs = jobs

    classifier.fit(values)

    if isinstance(classifier, LocalOutlierFactor):
        scores = classifier.negative_outlier_factor_
    else:
        scores = classifier.score_samples(values)

    return scores


def fetch_anomalies(scores, number=40):
    """Fetch indecies with lowest scores.

    Parameters
    ----------
    scores: Array to analyse.
    number: Number of anomalies to derive.

    Returns
    -------
    Index of anomalies.
    """
    index = np.argsort(scores)
    return index[:number]


def common_intersections(classifiers, values, n_outliers=40, adjacencies=5, iterations=1, use_tqdm=True):
    """Plot the curve of common intersections. That's the curve of estimated number of common results
    for successive models. Helps for choosing the classifier parameters.

    Parameters
    ----------
    classifiers: List of classifiers to run.
    values: Data to run classifiers on.
    n_outliers: Number of outliers to drive.
    adjacencies: Number of successive classifier results to intersect.
    iterations: Number of iterations to perform for estimating common intersections.
    use_tqdm: Whether use the progress bar for information.

    Returns
    -------
    Tuple of two lists. The first is the estimated number of common intersections (mean).
    The second is the RMS error of estimation.
    """
    range_fun = tnrange if use_tqdm else range

    if iterations == 1:
        indices = [None] * len(classifiers)
        for i in range_fun(len(classifiers)):
            index, _ = run_classifier(classifiers[i], values)
            indices[i] = set(index[0:n_outliers])

        volumes = np.empty(len(classifiers) - adjacencies + 1)
        for i in range(len(volumes)):
            intersection = set.intersection(*indices[i:i + adjacencies])
            volumes[i] = len(intersection)

        return volumes, np.zeros(volumes.shape)
    else:
        all_volumes = np.empty((len(classifiers) - adjacencies + 1, iterations))
        for i in range_fun(iterations):
            all_volumes[:, i], _ = common_intersections(classifiers, values, n_outliers, adjacencies, use_tqdm=use_tqdm)

        return np.mean(all_volumes, axis=1), np.std(all_volumes, axis=1)
