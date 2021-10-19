#!/usr/bin/env python

import click
import logging
import os
import sys
import webbrowser

import numpy as np
import pandas as pd

from ad_examples.common.utils import configure_logger
from ad_examples.common.gen_samples import get_synthetic_samples
from ad_examples.aad import aad_globals
from ad_examples.aad.aad_globals import (
    AAD_IFOREST, IFOR_SCORE_TYPE_NEG_PATH_LEN, HST_LOG_SCORE_TYPE, AAD_HSTREES, RSF_SCORE_TYPE,
    AAD_RSFOREST, INIT_UNIF, AAD_CONSTRAINT_TAU_INSTANCE, QUERY_DETERMINISIC, ENSEMBLE_SCORE_LINEAR,
    AadOpts
)
from ad_examples.aad.aad_support import get_aad_model
from ad_examples.aad.forest_aad_detector import is_forest_detector
from ad_examples.aad.forest_description import CompactDescriber, MinimumVolumeCoverDescriber, BayesianRulesetsDescriber, get_region_memberships
from ad_examples.aad.query_model import Query

from zwad.ad.transformation import transform_features as transform_lc_features
from zwad.utils import load_data

import onnx
from aad2onnx.convert import to_onnx

"""
A simple no-frills demo of how to use AAD in an interactive loop.

To execute:
pythonw -m aad.demo_aad
"""

logger = logging.getLogger(__name__)


class ResultsSink(object):
    def __init__(self, anomalies_filename, answers_filename):
        self._anomalies = open(anomalies_filename, mode="w", encoding="utf-8")

        self._answers = open(answers_filename, mode="w", encoding="utf-8")
        self._answers.write("oid, is_anomaly\n")

    def _handle_anomaly(self, name, decision):
        if not decision:
            return

        self._anomalies.write("{}\n".format(name))
        self._anomalies.flush()

    def _handle_answer(self, name, decision):
        self._answers.write("{}, {:b}\n".format(name, decision))
        self._answers.flush()

    def __call__(self, *args, **kwargs):
        self._handle_anomaly(*args, **kwargs)
        self._handle_answer(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self._anomalies.close()
        self._answers.close()

        return False


def load_answers(filename):
    df = pd.read_csv(filename)

    return df["oid"], df["is_anomaly"]


class InteractiveExpert(object):
    def __init__(self):
        try:
            self._browser = webbrowser.get()
        except webbrowser.Error:
            self._browser = None

    def evaluate(self, name):
        url = "https://ztf.snad.space/dr4/view/{}".format(name)
        if self._browser is not None:
            self._browser.open_new_tab(url)
        else:
            click.echo("Check {} for details".format(url))
        result = click.confirm("Is {} anomaly?".format(name))
        return result


class AnswersFileExpert(object):
    def __init__(self, filenames, spare_expert=None):
        dicts = [dict(zip(*load_answers(f))) for f in filenames]
        self._answers = dict([x for d in dicts for x in d.items()])
        self.spare_expert = spare_expert

    def evaluate(self, name):
        answer = self._answers.get(name)
        if answer is None and self.spare_expert is not None:
            return self.spare_expert.evaluate(name)
        return answer

class AllYesExpert(object):
    def __init__(self):
        pass

    def evaluate(self, name):
        return True


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
            "--resultsdir=./data/aad",
            "--log_file=./data/aad/aad.log",
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


def get_expert(opts):
    answers_filenames = opts.answers

    if opts.yes is True:
        return AllYesExpert()

    if answers_filenames is None:
        return InteractiveExpert()

    spare_expert = None
    if not opts.non_interactive:
        spare_expert = InteractiveExpert()

    return AnswersFileExpert(answers_filenames, spare_expert=spare_expert)


def detect_anomalies_and_describe(x, names, opts):
    rng = np.random.RandomState(opts.randseed)

    basename, _ = os.path.splitext(os.path.basename(opts.feature[0]))
    anomalies_filepath = os.path.join(opts.resultsdir, 'anomalies_{}.txt'.format(basename))
    answers_filename = os.path.join(opts.resultsdir, 'answers_{}.csv'.format(basename))

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
    with ResultsSink(anomalies_filepath, answers_filename) as sink:
        while len(queried) < opts.budget:
            ordered_idxs, anom_score = model.order_by_score(x_transformed)
            qx = qstate.get_next_query(ordered_indexes=ordered_idxs,
                                       queried_items=queried)
            queried.extend(qx)
            for xi in qx:
                yes = np.array(expert.evaluate(names[xi]), dtype=np.int64)
                y_labeled[xi] = yes
                if yes == 1:
                    ha.append(xi)
                else:
                    hn.append(xi)

                sink(names[xi], yes)

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
    
    return model, x_transformed, queried, ridxs_counts, region_extents


def get_aad_option_list():
    parser = aad_globals.get_aad_option_list()
    parser.add_argument('--answers', metavar='FILENAME', action='append', help='answers.csv file for AnswersFileExpert')
    parser.add_argument('--oid', metavar='FILENAME', action='append', help='Filepath to oid.dat', required=True)
    parser.add_argument('--feature', metavar='FILENAME', action='append', help='Filepath to feature.dat', required=True)
    parser.add_argument('--feature-names', metavar='FILENAME', help='Name of the file with feature names, one name per line')
    parser.add_argument('--transform', action='store_true', help='Data transformation using nonlinear functions')
    parser.add_argument('-n', '--non_interactive', action='store_true', help='Suppress InteractiveExpert')
    parser.add_argument('-y', '--yes', action='store_true', help='Select \'yes\' for every choise')
    parser.add_argument('--onnx_file', metavar='FILENAME', help='Dump trained model to ONNX format', required=False)
    return parser


class ZTFAadOpts(AadOpts):
    def __init__(self, args):
        super(ZTFAadOpts, self).__init__(args)
        self.oid = args.oid
        self.feature = args.feature
        self.answers = args.answers
        self.non_interactive = args.non_interactive
        self.yes = args.yes


def get_aad_command_args(argv, debug=False, debug_args=None):
    if debug_args is None:
        debug_args = []

    parser = get_aad_option_list()

    if argv:
        unparsed_args = argv[:1]
    else:
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

def main(argv=None):
    # Prepare the aad arguments. It is easier to first create the parsed args and
    # then create the actual AadOpts from the args
    args = get_aad_command_args(argv=argv, debug=True, debug_args=get_debug_args())
    try:
        os.makedirs(args.resultsdir)
    except OSError:
        pass
    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)))
    except OSError:
        pass
    configure_logger(args)

    opts = ZTFAadOpts(args)
    logger.debug(opts.str_opts())

    np.random.seed(opts.randseed)

    names, features = load_data(opts.oid, opts.feature)

    if args.transform:
        if args.feature_names is None:
            raise ValueError('--feature-names must be specified when --transform is enabled')
        with open(args.feature_names) as fh:
            feature_names = fh.read().split()
        transform_lc_features(features, feature_names)

    model, *_ = detect_anomalies_and_describe(features, names, opts)
    if args.onnx_file is not None:
        model_onnx = to_onnx(model, features)
        onnx.save_model(model_onnx, args.onnx_file)

def execute_from_commandline(argv=None):
    main(argv)

if __name__ == "__main__":
    main()

