import numpy as np
import json
import os
import argparse
import logging
import time
import pickle
import random
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate

os.environ["PATH"] += os.pathsep + "/usr/local/bin"
os.environ["PATH"] += os.pathsep + "/Library/Tex/texbin"

import matplotlib.pyplot as plt
import matplotlib as mpl

from collections import Counter
from sklearn.model_selection import ParameterGrid

import sys

sys.path.append(".")

from src.config import config as cfg
from src.utils import utils
from src.utils import logging

logger = logging.get_logger("./logs")

def evaluate_gntk_cv_run(config, results_dir):

    dataset = results_dir.split("/")[-1]
    exp_detail_path = results_dir.replace(f"{config.exp_path}/", "")
    out_dir = f"{config.eval_path}/{exp_detail_path}"

    utils.make_dirs_checked(out_dir)

    iteration_dirs = []
    for dir in os.listdir(results_dir):
        if dir.split("_")[0][:-1] == "iteration":
            iteration_dirs.append(dir)
    iteration_dirs.sort()

    cv_res = {
        "iteration_accuracy": [],
        "overall_accuracy_mean": 0.0,
        "overall_accuracy_std": 0.0,
        "iterations": {},
    }
    for i, d in enumerate(iteration_dirs):
        with open(f"{results_dir}/{d}/cv_results.txt", "r") as f:
            res = json.load(f)
        cv_res["iterations"][str(i)] = res
        cv_res["iteration_accuracy"].append(res["iteration_accuracy_mean"])
    cv_res["overall_accuracy_mean"] = np.mean(cv_res["iteration_accuracy"])
    cv_res["overall_accuracy_std"] = np.std(cv_res["iteration_accuracy"])

    # out_dir = os.path.join(results_dir, 'analysis')
    # if not os.path.isdir(out_dir):
    #     os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/cv_results.txt", "w") as f:
        json.dump(cv_res, f, indent=4)

    best_params = []
    for iteration in cv_res["iterations"].keys():
        for fold in cv_res["iterations"][iteration]["folds"].keys():
            best_params_tmp = cv_res["iterations"][iteration]["folds"][fold][
                "best_parameters"
            ]
            best_params.append(
                f"{best_params_tmp['K']}_normalize{best_params_tmp['normalize']}"
            )
    best_params_count = dict(Counter(best_params))

    with open(f"{out_dir}/best_param_count.txt", "w") as f:
        json.dump(best_params_count, f, indent=4)

    test_indices = {}
    test_predictions = {}
    K_validation_accuracy = {}
    for K in cv_res["iterations"]["0"]["folds"]["0"]["inner_res"]["K_results"].keys():
        for norm in ["True", "False"]:
            K_validation_accuracy[f"{K}_normalize{norm}"] = []

    for iteration, iteration_dict in cv_res["iterations"].items():
        test_indices[iteration] = {}
        test_predictions[iteration] = {}
        for fold, fold_dict in iteration_dict["folds"].items():

            test_indices[iteration][fold] = fold_dict["test_indices"]
            test_predictions[iteration][fold] = fold_dict["y_pred"]

            for K, res in fold_dict["inner_res"]["K_results"].items():
                for norm in ["True", "False"]:
                    key = f"{K}_normalize{norm}"
                    val = res["normalization"][norm]["best_accuracy_mean"]
                    K_validation_accuracy[key].append(val)

    with open(f"{out_dir}/test_indices.txt", "w") as f:
        json.dump(test_indices, f, indent=4)

    with open(f"{out_dir}/test_predictions.txt", "w") as f:
        json.dump(test_predictions, f, indent=4)

    score = f'{np.round(cv_res["overall_accuracy_mean"]*100,2)} ± {np.round(cv_res["overall_accuracy_std"]*100,2)}'

    with open(f"{out_dir}/{score}", "w") as f:
        f.write(score)

    dict_list = []
    best_non_norm = {"K_param": None, "accuracy": 0.0}

    params_dict = {
        "L": range(1, 15),
        "R": range(1, 4),
        "scale": ["uniform", "degree"],
        "jk": [0, 1],
        "norm": ["True", "False"],
    }
    param_grid = ParameterGrid(params_dict)

    for params in param_grid:
        L = params["L"]
        R = params["R"]
        scale = params["scale"]
        jk = params["jk"]
        norm = params["norm"]

        key = f"L{L}_R{R}_scale{scale}_jk{jk}_normalize{norm}"
        res = K_validation_accuracy[key]
        count = 0.0
        try:
            count += best_params_count[key]
        except:
            pass

        res_acc_mean = np.mean(res)
        res_acc_std = np.std(res)
        dict_list.append(
            {
                "L": L,
                "R": R,
                "scale": scale,
                "jk": jk,
                "norm": norm,
                "acc_mean": res_acc_mean,
                "acc_std": res_acc_std,
                "best_count": count,
            }
        )
        if norm == "False":
            if res_acc_mean > best_non_norm["accuracy"]:
                best_non_norm["K_param"] = key
                best_non_norm["accuracy"] = res_acc_mean

    K_validation_df = pd.DataFrame(dict_list)
    K_validation_df.to_csv(f"{out_dir}/K_validation_df.csv", index=False, sep=",")

    with open(f"{out_dir}/best_non_norm_params.txt", "w") as f:
        json.dump(best_non_norm, f)


def evaluate_gin_run(config, results_dir):

    dataset = results_dir.split("/")[-1]
    exp_detail_path = results_dir.replace(f"{config.exp_path}/", "")
    out_dir = f"{config.eval_path}/{exp_detail_path}"

    utils.make_dirs_checked(out_dir)

    iteration_accuracy = []
    test_indices = {}
    test_predictions = {}

    for iteration in range(10):

        fold_accuracy = []
        test_indices[iteration] = {}
        test_predictions[iteration] = {}

        for fold in range(10):

            with open(f'{results_dir}/rep{iteration}_fold{fold}.txt', 'r') as f:
                fold_file = f.read().split('\n')[:-1]

                fold_acc_curve = [float(item.split(' ')[2]) for item in fold_file]
                fold_accuracy.append(fold_acc_curve)

            fold_indices = np.loadtxt(
                f'{results_dir}/rep{iteration}_fold{fold}_test_indices.txt',
                delimiter=",",
                dtype=float
            ).round().astype(int)
            test_indices[iteration][fold] = fold_indices.tolist()

            fold_predictions = np.loadtxt(
                f'{results_dir}/rep{iteration}_fold{fold}_test_predictions.txt',
                delimiter=",",
                dtype=float
            ).round().astype(int)
            test_predictions[iteration][fold] = fold_predictions.tolist()

        iteration_mean_accuracy = np.array(fold_accuracy).mean(axis=0)
        best_epoch = np.argmax(iteration_mean_accuracy)

        iteration_accuracy.append(iteration_mean_accuracy[best_epoch])

    overall_accuracy_mean = np.mean(iteration_accuracy)
    overall_accuracy_std = np.std(iteration_accuracy)
    score = f'{np.round(overall_accuracy_mean * 100, 2)} ± {np.round(overall_accuracy_std * 100, 2)}'

    with open(f'{out_dir}/{score}', 'w') as f:
        f.write(score)

    np.savetxt(
        os.path.join(out_dir, 'iteration_accuracies.txt'),
        iteration_accuracy,
        delimiter=","
    )

    with open(f'{out_dir}/test_indices.txt', 'w') as f:
        json.dump(test_indices, f)

    with open(f'{out_dir}/test_predictions.txt', 'w') as f:
        json.dump(test_predictions, f)


def score_strings(acc_mean, acc_std):
    acc_mean = np.round(acc_mean * 100, 2)
    acc_std = np.round(acc_std * 100, 2)
    score = f"{acc_mean:.2f} ± {acc_std:.2f}"
    score_latex = f"${acc_mean:.2f} \\pm {acc_std:.2f}$"
    return score, score_latex


if __name__ == "__main__":
    config = cfg.Config()

    logger.info("-------------------------------------")
    logger.info(f"Evaluating experiment (b)")
    logger.info("-------------------------------------")

    datasets = ["IMDBBINARY", "IMDBMULTI", "MUTAG", "NCI1", "PROTEINS", "PTC"]
    kernels = [
        "GNTK",
        "VH",
        "EH",
        "HGKWL_seed0",
        "HGKSP_seed0",
        "MLG",
        "MP",
        "SP",
        "WL",
        "WLOA",
        "GIN",
    ]

    # out_dir = f"{config.reporting_path}/b_benchmark"
    # utils.make_dirs_checked(out_dir)

    # evaluate gntk and gin
    logger.info("-------------------------------------")
    logger.info(f"Evaluating individual experiments")
    logger.info("-------------------------------------")
    for dataset in datasets:
        logger.info(f"Dataset -- {dataset}")
        
        # gntk
        logger.info(f"-- Evaluating GNTK CV experiment")
        res_dir = f"{config.exp_path}/GNTK/b.1/{dataset}/"
        evaluate_gntk_cv_run(config, res_dir)

        # gin
        logger.info(f"-- Evaluating GIN experiment")
        gin_path = f"{config.exp_path}/GIN/{dataset}"
        model_dir = [f for f in os.listdir(gin_path) if f[0]=="L"]
        gin_dir = os.path.join(gin_path, model_dir[0])
        evaluate_gin_run(config, gin_dir)

    predictions = {}
    best_params = {}
    benchmark_table = pd.DataFrame()
    benchmark_table["datasets"] = datasets

    for K in kernels:
        predictions[K] = {dataset: [] for dataset in datasets}
        benchmark_table[K] = 0

        if not K == "GIN":
            best_params[K] = {dataset: None for dataset in datasets}

    benchmark_table_latex = benchmark_table.copy()

    logger.info("-------------------------------------")
    logger.info(f"Collecting results of individual experiments")
    logger.info("-------------------------------------")
    for dataset in datasets:
        logger.info(f"Dataset -- {dataset}")
        for K in kernels:
            logger.info(f"-- {K}")

            # collect GNTK first
            if K == "GNTK":
                # try:
                path = f"{config.eval_path}/GNTK/b.1/{dataset}"

                with open(f"{path}/test_predictions.txt", "r") as f:
                    preds = json.load(f)

                for iteration in range(10):
                    for fold in range(10):
                        predictions[K][dataset] += preds[str(iteration)][str(fold)]

                best = None
                best_count = 0
                with open(f"{path}/best_param_count.txt", "r") as f:
                    best_param_counts = json.load(f)

                for k, v in best_param_counts.items():
                    if v > best_count:
                        best = k
                        best_count = v

                best_params[K][dataset] = best

                with open(f"{path}/cv_results.txt", "r") as f:
                    cv_res = json.load(f)

                gntk_acc_mean = cv_res["overall_accuracy_mean"]
                gntk_acc_std = cv_res["overall_accuracy_std"]

                score, score_latex = score_strings(gntk_acc_mean, gntk_acc_std)

                benchmark_table.loc[benchmark_table.datasets == dataset, K] = score
                benchmark_table_latex.loc[
                    benchmark_table.datasets == dataset, K
                ] = score_latex

            elif (K != "GNTK") & (K != "GIN"):
                dataset_alt = dataset

                path = f"{config.exp_path}/Graphkernels"

                try:
                    with open(f"{path}/{dataset_alt}_{K}.json", "r") as f:
                        cv_res = json.load(f)

                    acc_list = []

                    hyperparams = (
                        cv_res["iterations"][str(iteration)]["folds"][str(fold)][
                            "kernels"
                        ][K]["best_model"]["K"]
                        != "K"
                    )
                    best_param_counts = {}

                    for iteration in range(10):
                        fold_acc_list = []
                        for fold in range(10):
                            fold_acc_list.append(
                                cv_res["iterations"][str(iteration)]["folds"][
                                    str(fold)
                                ]["kernels"][K]["accuracy"]
                            )
                            predictions[K][dataset] += cv_res["iterations"][
                                str(iteration)
                            ]["folds"][str(fold)]["kernels"][K]["y_pred"]

                            param = cv_res["iterations"][str(iteration)]["folds"][
                                str(fold)
                            ]["kernels"][K]["best_model"]["K"]

                            if param in best_params.keys():
                                best_param_counts[param] += 1
                            else:
                                best_param_counts[param] = 1

                        acc_list.append(np.mean(fold_acc_list))

                    if hyperparams:

                        best = None
                        best_count = 0

                        for k, v in best_param_counts.items():
                            if v > best_count:
                                best = k
                                best_count = v

                        best_params[K][dataset] = best

                    else:
                        best_params[K][dataset] = False

                    gk_acc_mean = np.mean(acc_list)
                    gk_acc_std = np.std(acc_list)

                    score, score_latex = score_strings(gk_acc_mean, gk_acc_std)

                    benchmark_table.loc[benchmark_table.datasets == dataset, K] = score
                    benchmark_table_latex.loc[
                        benchmark_table.datasets == dataset, K
                    ] = score_latex

                except:
                    best_params[K][dataset] = "NA"
                    predictions[K][dataset] = "NA"
                    benchmark_table.loc[benchmark_table.datasets == dataset, K] = "NA"

            # collect GIN results
            elif K == "GIN":
                gin_path = f"{config.eval_path}/GIN/{dataset}"
                model_dir = [f for f in os.listdir(gin_path) if f[0]=="L"]
                path = f"{config.eval_path}/GIN/{dataset}/{model_dir[0]}"
                gin_acc_list = np.loadtxt(f"{path}/iteration_accuracies.txt")
                with open(f"{path}/test_predictions.txt", "r") as f:
                    preds = json.load(f)

                for iteration in range(10):
                    for fold in range(10):
                        predictions[K][dataset] += preds[str(iteration)][str(fold)]

                gin_acc_mean = np.mean(gin_acc_list)
                gin_acc_std = np.std(gin_acc_list)

                score, score_latex = score_strings(gin_acc_mean, gin_acc_std)

                benchmark_table.loc[benchmark_table.datasets == dataset, K] = score
                benchmark_table_latex.loc[
                    benchmark_table.datasets == dataset, K
                ] = score_latex

    logger.info("Results --")
    logger.info(f"{benchmark_table.to_string()}")

    logger.info("Storing results")
    with open(f"{config.reporting_path}/exp_b_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    with open(f"{config.reporting_path}/exp_b_predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)

    benchmark_table.to_csv(f"{config.reporting_path}/exp_b_results_table.csv", index=False)
    with open(f"{config.reporting_path}/exp_b_results_table_md.txt", "w") as f:
        print(tabulate(benchmark_table, tablefmt="pipe", headers="keys"), file=f)
