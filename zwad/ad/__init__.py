import argparse
import warnings
import sys, os
from pathlib import Path
import numpy as np

from abc import ABC, abstractmethod

from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from coniferest.isoforest import IsolationForest

from zwad.ad.preprocess import scale_values
from zwad.ad.postprocess import save_anomaly_list
from zwad.ad.util import run_classifier, fetch_anomalies
from zwad.ad.transformation import parse_feature_name, identical, period_norm, period_norm_inv, transform_direct, \
    transform_inverse, transform_features


class BaseAnomalyDetector(ABC):
    classifiers = {
        'gmm': GaussianMixture(n_components=10, covariance_type='spherical', n_init=15),
        'svm': OneClassSVM(gamma='scale', nu=0.02, kernel='rbf'),
        'lof': LocalOutlierFactor(n_neighbors=100, contamination=0.02, algorithm='kd_tree', metric='euclidean'),
        'iso': IsolationForest(n_subsamples=1024, n_trees=3000),
    }

    def __init__(self, args=None):
        self.parser = self._make_parser()
        self.args = self.parser.parse_args(args=args)

        # Set the OOM (un)protection, if needed
        if self.args.kmp != 0:
            kill_me_please(self.args.kmp)

        # Load datasets
        self.names, self.values = self._load_dataset()

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

        # In-place transformation of self.values
        if self.args.transform:
            if self.args.feature_names is None:
                raise ValueError('--feature-names must be specified when --transform is enabled')
            with open(self.args.feature_names) as fh:
                self.feature_names = fh.read().split()
            transform_features(self.values, self.feature_names)

        # Scaling
        if self.args.scale.startswith('pca') and len(self.args.scale) > 3:
            # Case of pca15, pca41 etc
            self.values = scale_values(self.values, algorithm='pca')
            components = int(self.args.scale[3:])
            self.values = self.values[:, :components]
        else:
            self.values = scale_values(self.values, algorithm=self.args.scale)

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

    @abstractmethod
    def _load_dataset(self):
        pass

    @staticmethod
    def _make_parser():
        parser = argparse.ArgumentParser(description='Run sklearn AD algorithm on ZTF data')
        parser.add_argument('-c', '--classifier', default='iso', type=str,
                            help='AD algorithm: iso, lof, gmm, svm. Defaults to iso.')
        parser.add_argument('-n', '--number', default=-1, type=int,
                            help='Use first n objects, -1 for using all the objects.')
        parser.add_argument('-j', '--jobs', default=None, type=int,
                            help='Cores usage. Defaults to 1.')
        parser.add_argument('-m', '--kill-me-please', metavar='KMP', dest='kmp',
                            default=0, type=int, choices=[-1, 0, 1],
                            help='Hint for the kernel what to do when memory ends: -1 (never kill me), 0 (default), 1 (always kill me).')
        parser.add_argument('-a', '--anomalies', default=40, type=int,
                            help='Number of anomalies to derive. Defaults to 40')
        parser.add_argument('-o', '--output', default=None,
                            help='Dump all the scores to the selected file.')
        parser.add_argument('-s', '--seed', default=42, type=int,
                            help='Fix the seed for reproducibility. Defaults to 42.')
        parser.add_argument('-k', '--scale', default='std', type=str,
                            help='Scale algorithm. One of minmax, std, pca. Default is std. '
                                 'The last one may have optional number of components, e.g. -k pca15.')
        parser.add_argument('-t', '--transform', action='store_true',
                            help='Data transformation using nonlinear functions.')
        return parser

    @classmethod
    def script(cls):
        cls().run()

class ZtfAnomalyDetector(BaseAnomalyDetector):
    def __init__(self, args=None):
        super(ZtfAnomalyDetector, self).__init__(args)

    @classmethod
    def _make_parser(cls):
        parser = super(ZtfAnomalyDetector, cls)._make_parser()
        parser.add_argument('--oid', help='Name of the file with object IDs. May be repeated.',
                            required=True, action='append')
        parser.add_argument('--feature', help='Name of the file with corresponding features. May be repeated.',
                            required=True, action='append')
        parser.add_argument('--feature-names', help='Name of the file with feature names, one name per line.')
        return parser

    def _load_dataset(self):
        # Check number of datasets
        if len(self.args.oid) != len(self.args.feature):
            raise ValueError('number of oid files should be the same as features files')

        # Read OIDs and features
        names_array_list = []  # list of arrays with object names
        values_array_list = []  # list of arrays with object values (features)
        for i in range(len(self.args.oid)):
            names_one_array = np.memmap(self.args.oid[i], mode='r', dtype=np.int64)
            names_array_list.append(names_one_array)
            values_one_array = np.memmap(self.args.feature[i], mode='r', dtype=np.float32).reshape(names_one_array.size, -1)
            values_array_list.append(values_one_array)

        names = np.concatenate(names_array_list, axis=0)
        values = np.concatenate(values_array_list, axis=0)

        return names, values


def kill_me_please(kmp=0):
    """
    Tune the process parameters to instruct the kernel what to do,
    when memory ends.
    """

    if sys.platform != 'linux':
        raise ValueError('do not know how to tune the OOM score in ' + {sys.platform})

    with open(Path('/', 'proc', str(os.getpid()), 'oom_score_adj'), 'w') as file:
        file.write(str(1000 * kmp))
