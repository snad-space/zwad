#!/usr/bin/env python

import os
import sys

import click
import numpy as np

from ad_examples.aad import aad_globals
from ad_examples.aad.aad_support import *
from ad_examples.aad.forest_description import CompactDescriber, MinimumVolumeCoverDescriber, \
    BayesianRulesetsDescriber, get_region_memberships

import pandas as pd

"""
A simple no-frills demo of how to use AAD in an interactive loop.

To execute:
pythonw -m aad.demo_aad
"""

logger = logging.getLogger(__name__)


def save_answers(filename, names, decisions):
    df = pd.DataFrame(data = {
        "SN": names,
        "is_anomaly": decisions
    })

    df.to_csv(filename, sep=",", encoding="utf-8", index=False)


def load_answers(filename):
    df = pd.read_csv(filename)

    return df["SN"], df["is_anomaly"]


def save_anomalies(filename, names):
    with open(filename, 'w') as f:
        f.write("\n".join(map(str, names)))


class InteractiveExpert(object):
    def __init__(self):
        pass

    def evaluate(self, name):
        result = click.confirm("Is {} anomaly?".format(name))
        return result


class AnswersFileExpert(object):
    def __init__(self, filename, spare_expert=None):
        self.filename = filename
        self.spare_expert = spare_expert
        self._answers = dict(zip(*load_answers(self.filename)))

    def evaluate(self, name):
        answer = self._answers.get(name)
        if answer is None and self.spare_expert is not None:
            return self.spare_expert.evaluate(name)
        return answer


def get_debug_args(budget=30, detector_type=AAD_IFOREST):
    # return the AAD parameters what will be parsed later
    return ["--resultsdir=./temp", "--randseed=42",
            "--reruns=1",
            "--detector_type=%d" % detector_type,
            "--forest_score_type=%d" %
            (IFOR_SCORE_TYPE_NEG_PATH_LEN if detector_type == AAD_IFOREST
             else HST_LOG_SCORE_TYPE if detector_type == AAD_HSTREES
             else RSF_SCORE_TYPE if detector_type == AAD_RSFOREST else 0),
            "--init=%d" % INIT_UNIF,  # initial weights
            "--withprior", "--unifprior",  # use an (adaptive) uniform prior
            # ensure that scores of labeled anomalies are higher than tau-ranked instance,
            # while scores of nominals are lower
            "--constrainttype=%d" % AAD_CONSTRAINT_TAU_INSTANCE,
            "--querytype=%d" % QUERY_DETERMINISIC,  # query strategy
            "--num_query_batch=1",  # number of queries per iteration
            "--budget=%d" % budget,  # total number of queries
            "--tau=0.03",
            # normalize is NOT required in general.
            # Especially, NEVER normalize if detector_type is anything other than AAD_IFOREST
            # "--norm_unit",
            "--forest_n_trees=100", "--forest_n_samples=256",
            "--forest_max_depth=%d" % (100 if detector_type == AAD_IFOREST else 7),
            # leaf-only is preferable, else computationally and memory expensive
            "--forest_add_leaf_nodes_only",
            "--ensemble_score=%d" % ENSEMBLE_SCORE_LINEAR,
            # "--bayesian_rules",
            "--resultsdir=../../data/aad",
            "--log_file=../../data/aad/aad.log",
            "--sn_list=../../data/tsne/tsne.9.csv",
            '--fig_dir=../../fig',
            "--debug"]


def describe_instances(x, instance_indexes, model, opts, interpretable=False):
    """ Generates compact descriptions for the input instances

    :param x: np.ndarray
        The instance matrix with ALL instances
    :param instance_indexes: np.array(dtype=int)
        Indexes for the instances which need to be described
    :param model: Aad
        Trained Aad model
    :param opts: AadOpts
    :return: tuple, list(map)
        tuple: (region indexes, #instances among instance_indexes that fall in the region)
        list(map): list of region extents where each region extent is a
            map {feature index: feature range}
    """
    if not is_forest_detector(opts.detector_type):
        raise ValueError("Descriptions only supported by forest-based detectors")

    # setup dummy y
    y = np.zeros(x.shape[0], dtype=np.int32)
    y[instance_indexes] = 1

    if interpretable:
        if opts.bayesian_rules:
            # use BayesianRulesetsDescriber to get compact and [human] interpretable rules
            describer = BayesianRulesetsDescriber(x, y=y, model=model, opts=opts)
        else:
            # use CompactDescriber to get compact and [human] interpretable rules
            describer = CompactDescriber(x, y=y, model=model, opts=opts)
    else:
        # use MinimumVolumeCoverDescriber to get simply compact (minimum volume) rules
        describer = MinimumVolumeCoverDescriber(x, y=y, model=model, opts=opts)

    selected_region_idxs, desc_regions, rules = describer.describe(instance_indexes)

    _, memberships = get_region_memberships(x, model, instance_indexes, selected_region_idxs)
    instances_in_each_region = np.sum(memberships, axis=0)
    if len(instance_indexes) < np.sum(instances_in_each_region):
        logger.debug("\nNote: len instance_indexes (%d) < sum of instances_in_each_region (%d)\n"
                     "because some regions overlap and cover the same instance(s)." %
                     (len(instance_indexes), int(np.sum(instances_in_each_region))))

    if rules is not None:
        rule_details = []
        for rule in rules:
            rule_details.append("%s: %d/%d instances" % (str(rule),
                                                         len(rule.where_satisfied(x[instance_indexes])),
                                                         len(instance_indexes)))
        logger.debug("Rules:\n  %s" % "\n  ".join(rule_details))

    return zip(selected_region_idxs, instances_in_each_region), desc_regions


def show_figure(sn, fig_dir, ext='.png'):
    from PIL import Image

    filename = sn + ext
    for root, dirs, files in os.walk(fig_dir):
        if filename in files:
            break
    else:
        return
    filepath = os.path.join(root, filename)
    img = Image.open(filepath)
    img.show()


def get_expert(opts):
    answers_filename = opts.answers

    if answers_filename is None:
        return InteractiveExpert()

    spare_expert = None
    if not opts.non_interactive:
        spare_expert = InteractiveExpert()

    return AnswersFileExpert(answers_filename, spare_expert=spare_expert)


def detect_anomalies_and_describe(x, names, opts):
    rng = np.random.RandomState(opts.randseed)

    expert = get_expert(opts)

    # prepare the AAD model
    model = get_aad_model(x, opts, rng)
    model.fit(x)
    model.init_weights(init_type=opts.init)

    # get the transformed data which will be used for actual score computations
    x_transformed = model.transform_to_ensemble_features(x, dense=False, norm_unit=opts.norm_unit)

    # populate labels as some dummy value (-1) initially
    y_labeled = np.ones(x.shape[0], dtype=int) * -1

    # at this point, w is uniform weight. Compute the number of anomalies
    # discovered within the budget without incorporating any feedback
    baseline_scores = model.get_score(x_transformed, model.w)
    baseline_queried = np.argsort(-baseline_scores)

    qstate = Query.get_initial_query_state(opts.qtype, opts=opts, budget=opts.budget)
    queried = []  # labeled instances
    ha = []  # labeled anomaly instances
    hn = []  # labeled nominal instances
    while len(queried) < opts.budget:
        ordered_idxs, anom_score = model.order_by_score(x_transformed)
        qx = qstate.get_next_query(ordered_indexes=ordered_idxs,
                                   queried_items=queried)
        queried.extend(qx)
        for xi in qx:
            if opts.show_figures:
                basename = names[xi].split('_', 1)[0]
                show_figure(basename, opts.fig_dir)
            yes = np.array(expert.evaluate(names[xi]), dtype=np.int64)
            y_labeled[xi] = yes
            if yes == 1:
                ha.append(xi)
            else:
                hn.append(xi)

        # incorporate feedback and adjust ensemble weights
        model.update_weights(x_transformed, y_labeled, ha=ha, hn=hn, opts=opts, tau_score=opts.tau)

        # most query strategies (including QUERY_DETERMINISIC) do not have anything
        # in update_query_state(), but it might be good to call this just in case...
        qstate.update_query_state()

    # generate compact descriptions for the detected anomalies
    ridxs_counts, region_extents = None, None
    if len(ha) > 0:
        ridxs_counts, region_extents = describe_instances(x, np.array(ha), model=model,
                                                          opts=opts, interpretable=True)
        logger.debug("selected region indexes and corresponding instance counts (among %d):\n%s" %
                     (len(ha), str(list(ridxs_counts))))
        logger.debug("region_extents: these are of the form [{feature_index: (feature range), ...}, ...]\n%s" %
                     (str(region_extents)))

    basename = '_'.join(os.path.splitext(os.path.basename(f))[0] for f in opts.sn_list)

    anomalies_filepath = os.path.join(opts.resultsdir, 'anomalies_{}.txt'.format(basename))
    save_anomalies(anomalies_filepath, names[ha])

    answers_filename = os.path.join(opts.resultsdir, 'answers_{}.csv'.format(basename))
    save_answers(answers_filename, names[queried], y_labeled[queried])
    
    return model, x_transformed, queried, ridxs_counts, region_extents


def get_aad_option_list():
    parser = aad_globals.get_aad_option_list()
    parser.add_argument('--sn_list', metavar='FILENAME', nargs='+', help='Filepath to tSNE SN list')
    parser.add_argument('--show_figures', action='store_true', help='Show SN light curves')
    parser.add_argument('--fig_dir', action='store', default='../../fig', help='Folder with SN light curves figures')
    parser.add_argument('--answers', metavar='FILENAME', action='store', help='answers.csv file for AnswersFileExpert')
    parser.add_argument('-n', '--non_interactive', action='store_true', help='Suppress InteractiveExpert')
    return parser


class SNAadOpts(AadOpts):
    def __init__(self, args):
        super(SNAadOpts, self).__init__(args)
        self.sn_list = args.sn_list
        self.show_figures = args.show_figures
        self.fig_dir = args.fig_dir
        self.answers = args.answers
        self.non_interactive = args.non_interactive


def get_aad_command_args(debug=False, debug_args=None):
    if debug_args is None:
        debug_args = []

    parser = get_aad_option_list()

    unparsed_args = sys.argv[1:]
    if debug:
        unparsed_args = debug_args + unparsed_args

    args = parser.parse_args(unparsed_args)

    if args.startcol < 1:
        raise ValueError("startcol is 1-indexed and must be greater than 0")
    if args.labelindex < 1:
        raise ValueError("labelindex is 1-indexed and must be greater than 0")

    # LODA arguments
    args.keep = None
    args.exclude = None
    args.sparsity = np.nan
    args.explain = False
    #args.ntop = 30 # only for explanations
    args.marked = []

    return args


def main():
    # Prepare the aad arguments. It is easier to first create the parsed args and
    # then create the actual AadOpts from the args
    args = get_aad_command_args(debug=True, debug_args=get_debug_args())
    try:
        os.makedirs(args.resultsdir)
    except OSError:
        pass
    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)))
    except OSError:
        pass
    configure_logger(args)

    opts = SNAadOpts(args)
    logger.debug(opts.str_opts())

    np.random.seed(opts.randseed)

    sn_data_dfs = [pd.read_csv(f) for f in args.sn_list]
    sn_name = np.hstack([df.iloc[:,0].values for df in sn_data_dfs])
    if all('claimedtype' in df.columns for df in sn_data_dfs):  # extrapol file
        if len(sn_data_dfs) > 1:  # numerous data files
            sn_name += np.hstack([['_{}'.format(''.join(df.columns[1:4]))] * len(df) for df in sn_data_dfs])
        lc_data = np.vstack([df.loc[:, 'g-20':'i+100'] for df in sn_data_dfs])
        lc_data_norm = np.amax(lc_data, axis=1).reshape(-1, 1)
        thetas = [np.array(df.loc[:,'log_likehood':'theta_8']) for df in sn_data_dfs]
        theta_data = np.concatenate(thetas, axis=0)
        theta_data_norm = np.amax(theta_data, axis=0) - np.amin(theta_data, axis=0)
        theta_data = theta_data / theta_data_norm
        sn_data = np.hstack([lc_data / lc_data_norm, -2.5 * np.log10(lc_data_norm), theta_data])
    elif any('claimedtype' in df.columns for df in sn_data_dfs):
        raise ValueError('All sn_list files should be either tsne* or extrapol*')
    else:  # only tSNE file
        sn_data = np.hstack([df.iloc[:, 1:].values for df in sn_data_dfs])

    # run interactive anomaly detection loop
    detect_anomalies_and_describe(sn_data, sn_name, opts)


if __name__ == "__main__":
    main()

