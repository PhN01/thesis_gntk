import numpy as np
import json
import pandas as pd
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

import os
os.environ["PATH"] += os.pathsep + '/usr/local/bin'
os.environ["PATH"] += os.pathsep + '/Library/Tex/texbin'
os.environ["PATH"]

import sys
sys.path.append(".")

from src.figures import Plot
from src.config import config as cfg


class time_complexity(Plot):

    def __init__(self):
        super(time_complexity, self).__init__()

        self.config = cfg.Config()
        
        self.data = {}
        self._get_data()
        self._get_plot_dims()

    def _get_data(self):

        with open(f"{self.config.exp_path}/timing/gram_timing_50samples.txt", "r") as f:
            times = json.load(f)

        times_norm = {}
        for dataset, kernel_times in times.items():
            min_time = np.min(list(kernel_times.values()))
            times_norm[dataset] = {}
            for k, v in kernel_times.items():
                times_norm[dataset][k] = v / min_time

        times_norm_df = pd.DataFrame(times_norm).transpose()

        datasets = times_norm_df.index
        kernels = times_norm_df.columns

        datasets_alt = ['IMDB-BINARY', 'IMDB-MULTI', 'MUTAG', 'NCI1', 'PROTEINS', 'PTC\_MR']

        self.data["times_norm_df"] = times_norm_df
        self.data["datasets"] = datasets
        self.data["kernels"] = kernels
        self.data["datasets_alt"] = datasets_alt

    def _get_plot_dims(self):
        self.plot_dims = (15,6)
    
    def plot(self):
        self._set_style()

        fig = plt.figure(figsize=self.plot_dims)

        bar_width = 0.1
        ax = fig.add_subplot(1,1,1)
        ymax = np.max(self.data["times_norm_df"].values,axis=(0,1))

        for i, dataset in enumerate(self.data["datasets"]):
            x = i + 1
            for j, kernel in enumerate(self.data["kernels"]):
                ax.bar(
                    x + j * bar_width,
                    np.log10(self.data["times_norm_df"][kernel][i]+1),
                    label=kernel if i == 0 else None,
                    width=bar_width,
                    color=self.colors[j]
                )
                
        ax.set_xticks(np.arange(1,7) + 2 * bar_width)
        ax.set_xticklabels(self.data["datasets_alt"], size=16)
        ax.set_xlim(0.5,7)

        ax.set_yticklabels(['$10^{'+str(i)+'}$' for i in range(5)], size=20)
        ax.set_yticks(range(0,5))
        ax.set_ylabel('log$_{10}$(relative CPU time + 1)', size=20, labelpad=10)

        # overall x text
        fig.text(0.5, -0.02, 'Datasets', ha='center', va='center', size=20)
            
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc = 'lower center', fontsize=16, bbox_to_anchor = (0,-0.15,1,1),
                    bbox_transform = plt.gcf().transFigure, borderaxespad=0.1, ncol=5)

        plt.tight_layout()
            
        plt.savefig(f'{self.config.reporting_path}/figures/time_complexity.png', bbox_inches='tight')


if __name__=="__main__":

    time_complexity().plot()