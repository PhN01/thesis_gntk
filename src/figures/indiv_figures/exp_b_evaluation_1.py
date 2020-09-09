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


class exp_b_evaluation_1(Plot):

    def __init__(self):
        super(exp_b_evaluation_1, self).__init__()

        self.config = cfg.Config()
        
        self.data = {}
        self._get_data()
        self._get_plot_dims()


    def _get_data(self):

        datasets = [
            'IMDB-BINARY',
            'IMDB-MULTI',
            'MUTAG',
            'NCI1',
            'PROTEINS',
            'PTC\_MR'
        ]
        kernels = [
            'GNTK', 'VH', 'EH', 'HGKWL_seed0', 'HGKSP_seed0', 'MLG',
            'MP', 'SP', 'WL', 'WLOA','GIN'
        ]

        kernel_rename = {
            'HGKWL_seed0': 'WL-HGK',
            'HGKSP_seed0': 'SP-HGK',
            'MLG':'MLK',
            'WLOA':'WL-OA'
        }

        df = pd.read_csv(f"{self.config.reporting_path}/exp_b_results_table.csv").transpose()
        df = df.rename(columns=df.iloc[0,:]).drop(['datasets'], axis=0)
        df = df.iloc[[1, 2, 5, 7, 4, 8, 9, 3, 6, 10, 0], :]

        df = df.replace(np.nan, 0)

        mean_df = df.copy()
        for kernel in kernels:
            mean_df.loc[kernel,:] = mean_df.loc[kernel,:].apply(lambda x: float(x[:5]) if not x == 0 else 0)
        mean_df = mean_df.rename(index=kernel_rename).replace(0, np.nan)

        std_df = df.copy()
        for kernel in kernels:
            std_df.loc[kernel,:] = std_df.loc[kernel,:].apply(lambda x: float(x[-4:]) if not x == 0 else 0)
        std_df = std_df.rename(index=kernel_rename).replace(0, np.nan)

        x = np.arange(len(kernels))+1

        self.data["mean_df"] = mean_df
        self.data["std_df"] = std_df
        self.data["datasets"] = datasets
        self.data["kernels"] = kernels
        self.data["x"] = x

    def _get_plot_dims(self):
        self.plot_dims = (12,10)
    
    def plot(self):
        self._set_style()
        self.fmtr = mpl.ticker.StrMethodFormatter('{x:.0f}')

        fig = plt.figure(figsize=self.plot_dims)
        x = np.arange(self.data["mean_df"].shape[0])+1

        for i, dataset in enumerate(self.data["datasets"]):
            max_std = np.nanmax(self.data["std_df"].values[:,i])
            ax = fig.add_subplot(6,1,i+1)
            for j, kernel in enumerate(self.data["mean_df"].index):
                ax.plot(
                    [j+1]*2,
                    [
                        self.data["mean_df"].values[j,i]-self.data["std_df"].values[j,i],
                        self.data["mean_df"].values[j,i]+self.data["std_df"].values[j,i]
                    ],
                    '-',
                    color='black',
                    label=kernel,
                    linewidth=1
                )
                ax.plot(
                    j+1,
                    self.data["mean_df"].values[j,i],
                    '.',
                    color='black',
                    label=kernel
                )
            
            ax.plot(
                [1,11],
                [self.data["mean_df"].values[-1,i]]*2,
                '--',
                color=self.colors[1],
                linewidth=1
            )
            ax.plot(
                [1,11],
                [np.nanmax(self.data["mean_df"].values[:-2,i])]*2,
                '--',
                color=self.colors[0],
                linewidth=1
            )
            
            ax.set_xticks(self.data["x"])
            ax.set_xticklabels(self.data["mean_df"].index if i == 5 else "", size=16)

            start, end = ax.get_ylim()
            ticks = np.arange(np.round((start - max_std)/10)*10,np.round((end + max_std)/10)*10,10).tolist()
            if np.max(ticks) != 100:
                ticks.append(np.max(ticks)+10)
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticks, size=16)
            ax.yaxis.set_label_position("right")
            ax.set_ylabel('\\textbf{'+dataset+'}',size=13, rotation=270, labelpad=20)

            plt.gca().yaxis.set_major_formatter(self.fmtr)

        handles = [
            mpl.lines.Line2D([0],[0],color=self.colors[1], linestyle='--'),
            mpl.lines.Line2D([0],[0],color=self.colors[0], linestyle='--')
        ]
        labels = [
            'GNTK performance',
            'Best established GK performance'
        ]
        fig.legend(
            handles, labels, loc = 'lower center', fontsize=16, bbox_to_anchor = (0,-0.1,1,1),
            bbox_transform = plt.gcf().transFigure, borderaxespad=0.1, ncol=2
        )

        fig.text(0.51, -0.02, 'Model', ha='center', va='center', rotation='horizontal', size=20)
        fig.text(-0.02, 0.5, 'Classification Accuracy (\%)', ha='center', va='center', rotation='vertical', size=20)
        plt.tight_layout(rect=(0.06, 0.06, 1, 1))
        plt.subplots_adjust(hspace=0)

        plt.tight_layout()
            
        plt.savefig(f'{self.config.reporting_path}/figures/exp_b_evaluation_1.png', bbox_inches='tight')


if __name__ == "__main__":
    exp_b_evaluation_1().plot()