#!/usr/bin/env python3

import pandas as pd
import numpy as np

import sys
import argparse

from sklearn.neighbors import NearestNeighbors

from zwad.utils import load_data

parser = argparse.ArgumentParser(description='Lookup for nearest neighbors')
parser.add_argument('--oid', metavar='FILENAME', action='append', help='Filepath to oid.dat', required=True)
parser.add_argument('--feature', metavar='FILENAME', action='append', help='Filepath to feature.dat', required=True)
parser.add_argument('--lookup', metavar='OID', action='append', help='OID to lookup for the neighbors', required=True)
parser.add_argument('--neighbors', metavar='NUMBER', action='store', help='A number of neighbors to look for', type=int, default=5)
parser.add_argument('--algorithm', metavar='ALGO', action='store', help='ball_tree or kd_tree', default='kd_tree')

def oid_to_index(oids, oid):
    index = dict([(o,n) for (n,o) in enumerate(oids)])
    return np.array([index[o] for o in oid])

def main(argv=None):
    if argv is None:
        argv = sys.argv

    args = parser.parse_args(argv[1:])
    oids, features = load_data(args.oid, args.feature)

    mean = features.mean(axis=0)
    std = features.std(axis=0)
    features = (features - mean) / std

    lookup_oids = np.array(args.lookup, dtype=oids.dtype)
    lookup = oid_to_index(oids, lookup_oids)

    nn = NearestNeighbors(algorithm=args.algorithm)
    nn.fit(features)
    neigh_dist, neigh_ind = nn.kneighbors(features[lookup, :], n_neighbors=args.neighbors, return_distance=True)

    pattern = np.repeat(lookup_oids.reshape(-1,1), repeats=args.neighbors, axis=1)
    neighbor = oids[neigh_ind]
    dist = neigh_dist

    res = pd.DataFrame.from_dict({"lookup": pattern.reshape(-1), "neighbor": neighbor.reshape(-1), "distance": dist.reshape(-1)})
    print(res.to_string())

def execute_from_commandline(argv=None):
    main(argv)

if __name__ == "__main__":
    main()
