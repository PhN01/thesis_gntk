"""
@author: Philipp Nikolaus
@date: 25.03.2020
"""

import time
import datetime
import argparse
import logging
import os
import json
import pickle
from timeit import default_timer as timer
from tqdm import tqdm
import numpy as np

import sys

sys.path.append(".")

import warnings
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._validation import _fit_and_score
from sklearn.svm import SVC

from joblib import Parallel
from joblib import delayed

from src.utils import utils
from src.config import config as cfg
from src.utils import logging
from src.data import data_loaders

"""
This script runs a grid search routine for a set of parameters
and kernel matrices. Depending on the arguments parsed, the cross
validation can be nested or non-nested.
The non-nested variant is generally not recommended because its
performance estimate is biased and too optimistic. Here, this is
only used for reproducing results.
The default is nested cross validation.

Example call:
bsub -n 20 -W 4:00 -R "rusage[mem=8096]" python kernel_cv_v2.py
--gram_dir ../../data/GNTK_paper/NCI1/gram_matrices/standard_params
--dataset NCI1 --out_dir ../../experiments --rep 1 --kfold 10 --fold_files 1
--C_num 120 --nested 0 --verbose 0 --min_scale_mat 1
"""


@ignore_warnings(category=ConvergenceWarning)
def grid_search_K(
    clf, K_param, K, param_grid, y, train_folds, test_folds, all_indices, norm_only=0
):
    """
    Grid search routine that searches for the optimal set of clf_parameters for a given
        kernel matrix. Allows to parallelize grid search across different kernel matrices.
        Function is used in parallel_grid_search_cv(). Searches across param_grid for given
        train and test folds.

    Args:
        clf (): classifier to fit
        K_param (): Description string of kernel matrix (key in kernel_matrices dict)
        K (): Kernel matrix
        param_grid (): Parameters for the grid search
        y (): Vector of the data labels
        train_folds (): list of lists with the training indices, len(train_folds)=n_folds
        test_folds (): set of all indices occuring in train and test folds
        all_indices ():
        norm_only ():

    Returns:
        returns a tuple of the K_param and the results_object with keys
        ['best_accuracy_mean','best_accuracy_std','best_parameters']
    """
    # initializes the result object
    K_results = {
        "best_accuracy_mean": 0.0,
        "best_accuracy_std": 0.0,
        "best_parameters": None,
        "normalization": {
            "True": {"best_accuracy_mean": 0.0, "best_accuracy_std": 0.0},
            "False": {"best_accuracy_mean": 0.0, "best_accuracy_std": 0.0},
        },
    }

    # This ensures that we *cannot* access the test indices,
    # even if we try :)
    K = K[all_indices, :][:, all_indices]

    if norm_only:
        norm_list = [True]
    else:
        norm_list = [True, False]

    for normalize in norm_list:
        # Iterate over all sets of parameters in param_grid
        for parameters in list(param_grid):

            # copy the kernel matrix, so we can potentially make
            # changes to it (normalize) without affecting the
            # original matrix
            K_tmp = K.copy()

            # Normalize the kernel matrix if True
            if normalize:
                K_diag = np.sqrt(np.diag(K_tmp))
                K_tmp /= K_diag[:, None]
                K_tmp /= K_diag[None, :]

            # Remove the parameter because it does not pertain to
            # the classifier below.
            # clf_parameters = {
            #     key: value for key, value in parameters.items() if key not in ["normalize"]
            # }
            # initialize list that will holds fold accuracies of
            # the given parameter set
            results_per_parameters = []

            # Iterate over all k folds in train/test folds
            for fold_index, (train_index, test_index) in enumerate(
                zip(train_folds, test_folds)
            ):

                # Fit clf on train_index using the
                # current parameter set and return the accuracy on
                # the test index
                accuracy, params = _fit_and_score(
                    clone(clf),
                    K_tmp,
                    y,
                    scorer=make_scorer(accuracy_score),
                    train=train_index,
                    test=test_index,
                    verbose=0,
                    parameters=parameters,
                    fit_params=None,  # No additional parameters for `fit()`
                    return_parameters=True,
                )
                # Append the accuracy to the results list so we can
                # calculate the mean later
                results_per_parameters.append(accuracy)

            # Calculate the mean of the current set of parameters
            # across all folds
            parameters_res = (
                np.mean(results_per_parameters),
                np.std(results_per_parameters),
            )

            # If the current set of parameters performed better than
            # the previous best performance, overwrite the results
            # in K_results
            if (
                parameters_res[0]
                > K_results["normalization"][str(normalize)]["best_accuracy_mean"]
            ):
                K_results["normalization"][str(normalize)][
                    "best_accuracy_mean"
                ] = parameters_res[0]
                K_results["normalization"][str(normalize)][
                    "best_accuracy_std"
                ] = parameters_res[1]

            if parameters_res[0] > K_results["best_accuracy_mean"]:
                K_results["best_accuracy_mean"] = parameters_res[0]
                K_results["best_accuracy_std"] = parameters_res[1]
                K_results["best_parameters"] = parameters.copy()
                K_results["best_parameters"]["normalize"] = normalize

    return K_param, K_results


def parallel_grid_search_cv(
    clf,
    train_folds,
    test_folds,
    all_indices,
    param_grid,
    kernel_matrices,
    labels,
    verbose=0,
    norm_only=0,
):
    """
    Parallel grid search routine that takes the list of all kernel
        matrices as input and searches for the best performing kernel
        matrix in a parallelized setup.

    Args:
        clf (): Classifier to fit
        train_folds (): list of lists with the training indices, len(train_folds)=n_folds
        test_folds (): list of lists with the test indices, len(test_folds)=n_folds
        all_indices (): set of all indices occuring in train and test folds
        param_grid (): Parameters for the grid search
        kernel_matrices (): Kernel matrices to check; each one of them
            is assumed to represent a different choice of parameter. They will
            *all* be checked iteratively by the routine.
        labels (): Vector of all labels of the datasets
        verbose (): Verbosity of the process. If verbose==1, tqdm is applied over kernel matrices
        norm_only (): If true, then normalize all kernel matrices by default

    Returns:
    Best classifier, i.e. the classifier with the best
        parameters. Needs to be refit prior to predicting labels on
        the test data set. Moreover, the best-performing matrix, in
        terms of the grid search, is returned. It has to be used in
        all subsequent prediction tasks. Additionally, the function
        also returns a results object of the grid search.
    """
    # only retain the labels of the indices given in
    # train_folds and test_folds
    y = labels[all_indices]

    # Initialize result objects to store results of grid search in
    best_clf = None
    best_K = None
    results = {
        "best_accuracy_mean": 0.0,
        "best_accuracy_std": 0.0,
        "best_parameters": {},
        "K_results": {},
    }

    # From this point on, `train_index` and `test_index` are supposed to
    # be understood *relative* to the input training indices.

    # case switch to allow for verbosity if required. Mostly for testing
    if verbose:

        # Parallelize search of best parameters for each kernel matrix
        # returns a list containing the tuple (kernel_description, results)
        # for each kernel matrix as items
        # TODO: add n_jobs as parsable parameter?
        joblib_output = Parallel(n_jobs=-1)(
            delayed(grid_search_K)(
                clf,
                K_param,
                K,
                param_grid,
                y,
                train_folds,
                test_folds,
                all_indices,
                norm_only=norm_only,
            )
            for K_param, K in tqdm(kernel_matrices.items())
        )
    else:

        # as above
        joblib_output = Parallel(n_jobs=-1)(
            delayed(grid_search_K)(
                clf,
                K_param,
                K,
                param_grid,
                y,
                train_folds,
                test_folds,
                all_indices,
                norm_only=norm_only,
            )
            for K_param, K in kernel_matrices.items()
        )

    # Iterate over the list of outputs and for each determine if the
    # kernel matrix performed better than any of the previously
    # evaluated
    for K_out in joblib_output:

        # write kernel matrix resutls to results object
        results["K_results"][K_out[0]] = K_out[1]

        # if current kernel matrix performed better than any of the
        # previous, overwrite results with current kernel matrix
        if K_out[1]["best_accuracy_mean"] > results["best_accuracy_mean"]:
            results["best_accuracy_mean"] = K_out[1]["best_accuracy_mean"]
            results["best_accuracy_std"] = K_out[1]["best_accuracy_std"]

            # extract the parameters of the best performing matrix and
            # initialize the best classifier accordingly
            results["best_parameters"] = K_out[1]["best_parameters"]
            # normalize is not a clf parameter, thus removing it
            clf_parameters = {
                k: v
                for k, v in K_out[1]["best_parameters"].items()
                if not k == "normalize"
            }
            best_clf = clone(clf).set_params(**clf_parameters)

            # add description of kernel matrix to best_parameters in
            # results object and retrieve the corresponding kernel
            # matrix to return
            results["best_parameters"]["K"] = K_out[0]
            best_K = kernel_matrices[results["best_parameters"]["K"]]
            if results["best_parameters"]["normalize"]:
                K_diag = np.sqrt(np.diag(best_K))
                best_K /= K_diag[:, None]
                best_K /= K_diag[None, :]
        else:
            pass

    return best_clf, best_K, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MUTAG", help="Name of the dataset")
    parser.add_argument(
        "--experiment", default="a.1", help="Experiment name in ['a.1','a.2','b.1']"
    )
    parser.add_argument(
        "--iteration_idx",
        default=0,
        type=int,
        help="Index of the cross validation iteration (repeated CV)",
    )
    parser.add_argument(
        "--verbose",
        default=0,
        type=int,
        help="Verbosity of the process. If 0 minimal information is shown during the process",
    )
    args = parser.parse_args()

    config = cfg.Config()
    cv_config = cfg.CVConfig(args.experiment)

    data_dir = f"{config.data_path}/{cv_config.data_source}/{args.dataset}"
    out_dir = f"{config.exp_path}/GNTK/{args.experiment}/{args.dataset}/iteration{args.iteration_idx}"

    assert not os.path.isfile(
        f"{out_dir}/cv_results.txt"
    ), "Job already finished. Stopping process."

    # create output directory if it does not exist yet
    utils.make_dirs_checked(out_dir)

    logger = logging.get_logger(out_dir)

    logger.info("================================")
    logger.info("Running kernel CV")
    logger.info("================================")
    logger.info("")
    logger.info(f"Experiment -- {args.experiment}")
    logger.info(f"Dataset -- {args.dataset}")
    logger.info(f"Iteration index -- {args.iteration_idx}")
    logger.info("")

    # load data
    # load all matrices in args.gram_dir and write them to a
    # dictionary with {gram_description: kernel_matrix}
    logger.info("Loading matrices...")
    gram_matrices, labels = data_loaders.load_gntk_matrices(
        source=cv_config.data_source,
        dataset=args.dataset,
        min_scale_mat=cv_config.min_scale_mat,
    )

    n_mat = len(gram_matrices)
    logger.info(f"Found {n_mat} matrices")

    n_graphs = len(labels)
    all_indices = np.arange(n_graphs)

    # initialize parameter grid for grid search. The number of C values
    # to search over is a tunable parameter, since Du et al. use num=120
    # but this is too expensive for repeated nested CV
    param_grid = {"C": np.logspace(-2, 4, num=cv_config.C_num)}
    if False:
        norm_only = True
    else:
        norm_only = False

    # Prepare time measurement
    start_time = timer()

    # case switch: if args.nested == True perform nested CV, otherwise
    # perform non-nested CV
    # making sure that if the fold files of the original GNTK paper are provided we only perform one iteration of CV
    if cv_config.fold_files:
        args.iteration_idx = 0

    if cv_config.nested:

        logger.info(
            f"Starting iteration {args.iteration_idx} of {cv_config.k_fold} fold nested CV."
        )

        # prepare results object for repeated nested CV
        cv_res = {
            "fold_accuracy": [],
            "iteration_accuracy_mean": 0.0,
            "iteration_accuracy_std": 0.0,
            "folds": {},
        }

        # create cross validation folds
        # case switch: if no fold files are provided, then use StratifiedKFold
        # to generate folds. Otherwise load the fold files
        if not cv_config.fold_files:

            cv = StratifiedKFold(
                n_splits=cv_config.k_fold,
                shuffle=True,
                random_state=42 + args.iteration_idx,  # TODO: make configurable?
            )

            # write train and test folds into separate lists such that
            # len(train_folds) = len(test_folds) = n_folds
            train_folds = []
            test_folds = []
            for fold_index, (train_index, test_index) in enumerate(
                cv.split(all_indices, labels)
            ):
                train_folds.append(train_index.tolist())
                test_folds.append(test_index.tolist())

        else:
            # set directory of the fold files
            fold_dir = f"{data_dir}/10fold_idx"
            try:
                # load the fold files in separate lists for train and test folds
                # such that len(train_folds) = len(test_folds) = n_folds
                train_folds = [
                    np.loadtxt(f"{fold_dir}/train_idx-{i}.txt").astype(int).tolist()
                    for i in range(1, 11)
                ]
                test_folds = [
                    np.loadtxt(f"{fold_dir}/test_idx-{i}.txt").astype(int).tolist()
                    for i in range(1, 11)
                ]
            except:
                raise FileNotFoundError(f"No fold files found in {fold_dir}.")

        # Iterate over the folds defined in train_folds and test_folds
        # Within each fold find the best performing model for the train_indices
        # and use the latter to predict the test_indices
        for fold_index, (train_indices, test_indices) in tqdm(
            enumerate(zip(train_folds, test_folds)), "Folds", total=cv_config.k_fold
        ):

            # create results dict for current fold and set alias
            fold_res = cv_res["folds"][fold_index] = {}

            # set inner loop cross validation folds to search best performing model
            # same logic as before when creating folds with StratifiedKFold
            # here, folds are created only from train_indices for model selection,
            # because we use train indices for prediction after model selection
            inner_cv = StratifiedKFold(
                n_splits=cv_config.inner_k_fold, shuffle=True, random_state=42
            )
            inner_train_folds = []
            inner_test_folds = []
            # here we use train_indices as input
            for fold_index, (train_index, test_index) in enumerate(
                inner_cv.split(train_indices, labels[train_indices])
            ):
                inner_train_folds.append(train_index)
                inner_test_folds.append(test_index)

            # initiate grid search model selection across all kernel matrices
            # and the entire param_grid using the inner folds.
            best_clf, best_K, inner_res = parallel_grid_search_cv(
                SVC(
                    class_weight="balanced"
                    if cv_config.balanced
                    else None,  # TODO: clarify use / careful when comparing results
                    kernel="precomputed",
                    max_iter=cv_config.max_iter,
                    probability=True,
                ),
                inner_train_folds,
                inner_test_folds,
                train_indices,
                ParameterGrid(param_grid),
                gram_matrices,
                labels,
                verbose=args.verbose,
                norm_only=norm_only,
            )

            # write the results of the grid search to the fold results object
            fold_res["inner_res"] = inner_res
            fold_res["best_parameters"] = inner_res["best_parameters"]

            # prepare the training data for retraining the best performing
            # model from the grid search on the entire train_indices and
            # and fit the model
            K_train = best_K[train_indices, :][:, train_indices]
            y_train = labels[train_indices]
            best_clf.fit(K_train, y_train)

            # prepare the test data and make predictions
            K_test = best_K[test_indices, :][:, train_indices]
            y_test = labels[test_indices]
            y_pred = best_clf.predict(K_test)
            y_score = best_clf.predict_proba(K_test)

            # calculate test accuracy and store results in fold results object
            accuracy = accuracy_score(y_test, y_pred)

            fold_res["test_accuracy"] = accuracy
            fold_res["y_pred"] = y_pred.tolist()
            fold_res["y_score"] = y_score.tolist()
            fold_res["train_indices"] = train_indices
            fold_res["test_indices"] = test_indices

            # write overall test accuracy to the fold-wise accuracy list of the
            # iteration results object
            cv_res["fold_accuracy"].append(accuracy)

        # after finishing all folds, calculate the accuracy mean and
        # stdev for the entire iteration
        cv_res["iteration_accuracy_mean"] = np.mean(cv_res["fold_accuracy"])
        cv_res["iteration_accuracy_std"] = np.std(cv_res["fold_accuracy"])

        cv_res["runtime"] = timer() - start_time

        with open(f"{out_dir}/cv_results.txt", "w") as f:
            json.dump(cv_res, f, indent=4)

    # case switch: if args.nested == False perform non-nested CV, otherwise
    # perform non-nested CV
    elif not cv_config.nested:

        logger.info(
            f"Starting iteration {args.iteration_idx} times repeated {cv_config.k_fold} fold non-nested CV."
        )

        # prepare results object for the non-nested CV
        cv_res = {
            "overall_accuracy_mean": 0.0,
            "overall_accuracy_std": 0.0,
            "best_parameters": None,
            "K_results": None,
        }

        # case switch: if no fold files are provided, then use StratifiedKFold
        # to generate folds. Otherwise load the fold files
        if not cv_config.fold_files:
            cv = StratifiedKFold(
                n_splits=cv_config.k_fold,
                shuffle=True,
                random_state=42,  # TODO: make configurable?
            )
            # write train and test folds into separate lists such that
            # len(train_folds) = len(test_folds) = n_folds
            train_folds = []
            test_folds = []
            for fold_index, (train_index, test_index) in enumerate(
                cv.split(all_indices, labels)
            ):
                train_folds.append(train_index)
                test_folds.append(test_index)

        else:
            # set directory of the fold files
            fold_dir = f"{data_dir}/10fold_idx"
            try:
                # load the fold files in separate lists for train and test folds
                # such that len(train_folds) = len(test_folds) = n_folds
                train_folds = [
                    np.loadtxt(f"{fold_dir}/train_idx-{i}.txt").astype(int).tolist()
                    for i in range(1, 11)
                ]
                test_folds = [
                    np.loadtxt(f"{fold_dir}/test_idx-{i}.txt").astype(int).tolist()
                    for i in range(1, 11)
                ]
            except:
                raise FileNotFoundError(f"No fold files found in {fold_dir}.")

        # initiate grid search across all kernel matrices and the entire
        # param_grid using the inner folds. Since we perform non-nested
        # CV, we just report the output of this grid-search
        _, _, inner_res = parallel_grid_search_cv(
            SVC(
                class_weight="balanced" if cv_config.balanced else None,
                kernel="precomputed",
                max_iter=cv_config.max_iter,
                probability=True,
            ),
            train_folds,
            test_folds,
            all_indices,
            ParameterGrid(param_grid),
            gram_matrices,
            labels,
            verbose=1,
        )

        # write results of grid search to the results object
        cv_res["K_results"] = inner_res["K_results"]
        cv_res["best_parameters"] = inner_res["best_parameters"]
        cv_res["overall_accuracy_mean"] = inner_res["best_accuracy_mean"]
        cv_res["overall_accuracy_std"] = inner_res["best_accuracy_std"]

        # also store folds and runtime
        cv_res["train_folds"] = train_folds
        cv_res["test_folds"] = test_folds
        cv_res["runtime"] = timer() - start_time

        with open(f"{out_dir}/cv_results.txt", "w") as f:
            json.dump(cv_res, f, indent=4)

    logger.info("Experiment finished.")

