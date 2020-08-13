import numpy as np
import json
import os
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate

import sys
sys.path.append('.')

from src.config import config as cfg
from src.utils import utils
from src.utils import logging

logger = logging.get_logger("./logs")

def score_strings(acc_mean, acc_std):
    acc_mean = np.round(acc_mean * 100,2)
    acc_std = np.round(acc_std * 100,2)
    score = f'{acc_mean:.2f} ± {acc_std:.2f}'
    score_latex = f'${acc_mean:.2f} \\pm {acc_std:.2f}$'
    return score, score_latex


if __name__ == "__main__":
    config = cfg.Config()
    logger.info("-------------------------------------")
    logger.info("Evaluating experiment (a)")
    logger.info("-------------------------------------")

    datasets = ['IMDBBINARY','IMDBMULTI','MUTAG','NCI1','PROTEINS','PTC']

    utils.make_dirs_checked(config.reporting_path)

    replication_df = pd.DataFrame(np.zeros((3,6)))
    orig_results = ['76.9 ± 3.6', '52.8 ± 4.6', '90.0 ± 8.5', '84.2 ± 1.5', '75.6 ± 4.2', '67.9 ± 6.9']
    replication_df.iloc[0,:] = orig_results

    for i, dataset in tqdm(enumerate(datasets)):

        # (a.1)
        res_dir = f'{config.exp_path}/GNTK/a.1/{dataset}/iteration0'

        with open(f'{res_dir}/cv_results.txt', 'r') as f:
            cv_res = json.load(f)

        acc_mean = cv_res['overall_accuracy_mean']
        acc_std = cv_res['overall_accuracy_std']

        replication_df.iloc[1,i], _ = score_strings(acc_mean, acc_std)

        # (a.2)
        res_dir = f'{config.exp_path}/GNTK/a.2/{dataset}/iteration0'

        with open(f'{res_dir}/cv_results.txt', 'r') as f:
            cv_res = json.load(f)

        acc_mean = cv_res['iteration_accuracy_mean']
        acc_std = cv_res['iteration_accuracy_std']

        replication_df.iloc[2, i], _ = score_strings(acc_mean, acc_std)

    indices = ['Du et al.', '(a.1)', '(a.2)']
    columns = ['IMDB-BINARY', 'IMDB-MULTI', 'MUTAG', 'NCI1', 'PROTEINS', 'PTC_MR']
    replication_df = replication_df.rename(
        index={i: idx for i, idx in enumerate(indices)},
        columns={i: col for i, col in enumerate(columns)}
    )
    logger.info("Experiment results -- ")
    logger.info(f"{replication_df.to_string()}")

    replication_df.to_csv(f'{config.reporting_path}/exp_a_results_table.csv')

    with open(f'{config.reporting_path}/exp_a_results_table_md.txt', 'w') as f:
        print(tabulate(
            replication_df,
            tablefmt="pipe",
            headers="keys"
        ), file=f)