import os
import glob
import numpy as np
import pandas as pd

"""
Module for postprocessing AD results. AD results are
stored in CSV files and then assembled in DataFrame tables
for further querying.
"""


def load_ad_table(path, index_column='oid'):
    """Load one file with AD results.

    Parameters
    ----------
    path: Path to CSV file for loading.
    index_column: Name of the first column. Defaults to 'oid'. The second column's name is driven from filename.

    Returns
    -------
    DataFrame table.
    """
    _, filename = os.path.split(path)
    basename, _ = os.path.splitext(filename)
    return pd.read_csv(path, names=[index_column, basename], index_col=None)


def load_ad_tables_by_patterns(patterns, sorting='hits'):
    """Load all the file with AD results matching any patterns from the specified list.

    Parameters
    ----------
    patterns: Glob patterns of files for loading.
    sorting: Sorting preference:
        * 'hits' sorts by number of anomaly hits with different algos. (Default)
        * None for no sorting.

    Returns
    -------
    DataFrame table.
    """
    all_tables = []

    for pattern in patterns:
        one_pattern_tables = [load_ad_table(filename) for filename in glob.glob(pattern)]
        all_tables.append(merge_ad_tables(one_pattern_tables))

    table = merge_ad_tables(all_tables)

    # Sort the table, if needed
    if sorting == 'hits':
        index = table.isna().sum(axis=1).sort_values(kind='mergesort').index
        table = table.loc[index].reset_index(drop=True)
    elif sorting is None:
        pass
    else:
        raise ValueError('unknown sorting algorithm: {}'.format(sorting))

    return table


def merge_ad_tables(tables, index_column='oid'):
    """Merge the list of tables with AD results.

    Parameters
    ----------
    tables: list of AD results tables to load
    index_column: Column to index on. Defaults to 'oid'.

    Returns
    -------
    Merged DataFrame table.
    """
    tbl = pd.DataFrame({index_column: []})
    for t in tables:
        tbl = pd.merge(tbl, t, on=index_column, how='outer')

    return tbl


def extract_ad_subtable(table, value_columns):
    """Extract subtable from the AD results table.
    The results are sorted in the way suitable for expert analysis.

    Parameters
    ----------
    table: Source table to query from.
    value_columns: Which columns should we preserve.

    Returns
    -------
    Extracted subtable.
    """
    columns = ['oid'] + value_columns
    subtable = table.loc[:, columns].dropna(thresh=2).sort_values(by=value_columns).reset_index(drop=True)
    sorted_index = subtable.iloc[:, 1:].isna().sum(axis=1).values.argsort(kind='stable')
    sorted_subtable = subtable.loc[sorted_index].reset_index(drop=True)
    return sorted_subtable


def save_anomaly_list(filename, names, scores):
    """Save the list of anomalies in a uniform way.

    Parameters
    ----------
    filename: Name of the file to save to.
    names: Array of anomaly names.
    scores: Array of anomaly scores. Lesser score means more anomalous object.

    Returns
    -------
    None
    """
    table = pd.concat((pd.Series(names), pd.Series(scores)), axis=1)
    table.to_csv(filename, header=False, index=False)


def load_expert_table(filename):
    """
    Load table with expert analysis of the anomalies.

    Parameters
    ----------
    filename: Name of the file for loading.

    Returns
    -------
    Loaded table.
    """
    return pd.read_csv(filename, header=0, index_col=0)


def extract_anomaly_features(anomalies, oids, features):
    """
    Extract features of anomalies.

    Parameters
    ----------
    anomalies: 1D array of anomalies to extract features for.
    oids: Dataset oids, 1D array.
    features: Dataset features, 2D array.

    Returns
    -------
    Numpy array of features for anomalies.
    """
    d = dict(zip(oids, range(len(oids))))  # not very efficient, yes
    index = np.vectorize(lambda oid: d[oid])(anomalies)
    return features[index]
