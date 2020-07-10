import numpy as np

def load_data(oid_list, feature_list):
    filenames = zip(oid_list, feature_list)

    def load_single(oid_filename, feature_filename):
        oid     = np.memmap(oid_filename, mode='c', dtype=np.uint64)
        feature = np.memmap(feature_filename, mode='c', dtype=np.float32).reshape(oid.shape[0], -1)

        return oid, feature

    oids, features = zip(*[load_single(*f) for f in filenames])
    return np.concatenate(oids), np.vstack(features)
