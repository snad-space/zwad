import argparse
import warnings
import sys
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

from zwad.ad.preprocess import scale_values
from zwad.ad.postprocess import save_anomaly_list
from zwad.ad.util import run_classifier, fetch_anomalies


class ZtfAnomalyDetector:
    classifiers = {
        'gmm': GaussianMixture(n_components=10, covariance_type='spherical', n_init=15),
        'svm': OneClassSVM(gamma='scale', nu=0.02, kernel='rbf'),
        'lof': LocalOutlierFactor(n_neighbors=100, contamination=0.02, algorithm='kd_tree', metric='euclidean'),
        'iso': IsolationForest(max_samples=1000, contamination='auto', behaviour='new', n_estimators=1000),
    }

    def __init__(self):
        self.parser = self._make_parser()
        self.args = self.parser.parse_args()

        # Check number of datasets
        if len(self.args.oid) != len(self.args.features):
            raise ValueError('number of oid files should be the same as features files')

        # Read OIDs and features
        names_array_list = []  # list of arrays with object names
        values_array_list = []  # list of arrays with object values (features)
        for i in range(len(self.args.oid)):
            names_one_array = np.memmap(self.args.oid[i], dtype=np.int64)
            names_array_list.append(names_one_array)
            values_one_array = np.memmap(self.args.features[i], dtype=np.float32).reshape(names_one_array.size, -1)
            values_array_list.append(values_one_array)

        self.names = np.concatenate(names_array_list, axis=0)
        self.values = np.concatenate(values_array_list, axis=0)

        # Classifier
        self.classifier = self.classifiers[self.args.classifier]

        # Number of anomalies to derive
        self.n = self.args.anomalies

        # Fix the random seed
        self.seed = self.args.seed

        # Cut the data if needed
        n = self.args.number
        if n >= 0:
            self.names = self.names[:n]
            self.values = self.values[:n, :]

        # Scaling
        self.values = scale_values(self.values)

        # Jobs number
        self.jobs = self.args.jobs

        # Check NaNs
        self.check_nans()

    def check_nans(self):
        index = np.any(np.isnan(self.values), axis=0)
        if np.any(index):
            message = "Columns {} contain NaNs and are being ignored in AD analysis."
            warnings.warn(message.format(np.where(index)[0]), category=RuntimeWarning)
            self.values = self.values[:, ~index]

    def run(self):
        np.random.seed(self.seed)
        scores = run_classifier(self.classifier, self.values, jobs=self.jobs)
        index = fetch_anomalies(scores, number=self.n)
        save_anomaly_list(sys.stdout, self.names[index], scores[index])

        # Dump all the scores
        if self.args.output:
            scores.tofile(self.args.output)

    @staticmethod
    def _make_parser():
        parser = argparse.ArgumentParser(description='Run sklearn AD algorithm on ZTF data')
        parser.add_argument('-c', '--classifier', default='iso', type=str,
                            help='AD algorithm: iso, lof, gmm, svm. Defaults to iso.')
        parser.add_argument('-n', '--number', default=-1, type=int,
                            help='Use first n objects, -1 for using all the objects.')
        parser.add_argument('-j', '--jobs', default=None, type=int,
                            help='Cores usage. Defaults to 1.')
        parser.add_argument('-a', '--anomalies', default=40, type=int,
                            help='Number of anomalies to derive. Defaults to 40')
        parser.add_argument('-o', '--output', default=None,
                            help='Dump all the scores to the selected file.')
        parser.add_argument('-s', '--seed', default=42, type=int,
                            help='Fix the seed for reproducibility. Defaults to 42.')
        parser.add_argument('--oid', help='Name of the file with object IDs. May be repeated.',
                            required=True, action='append')
        parser.add_argument('--features', help='Name of the file with corresponding features. May be repeated.',
                            required=True, action='append')
        return parser

    @classmethod
    def script(cls):
        cls().run()

