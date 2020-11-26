import numpy as np
import pandas as pd


def load_data(oid_list, feature_list):
    filenames = zip(oid_list, feature_list)

    def load_single(oid_filename, feature_filename):
        oid     = np.memmap(oid_filename, mode='c', dtype=np.uint64)
        feature = np.memmap(feature_filename, mode='c', dtype=np.float32).reshape(oid.shape[0], -1)

        return oid, feature

    oids, features = zip(*[load_single(*f) for f in filenames])
    return np.concatenate(oids), np.vstack(features)


def latex_feature_names(path):
    """Gives mapping between feature code names and pretty LaTeX names"""
    df = pd.read_csv(path, index_col='short')
    latex = df['latex']
    return latex.to_dict()
