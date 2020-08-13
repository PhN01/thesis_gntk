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


class exp_a_evaluation_2(Plot):

    def __init__(self):
        super(exp_a_evaluation_2, self).__init__()

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
        rows = [
            'Du et al.',
            '(a.1)',
            '(a.2)'
        ]

        res_df = pd.read_csv(f"{self.config.reporting_path}/exp_a_results_table.csv", sep=",", index_col=0).rename(
            columns={str(i): datasets[i] for i in range(6)}, 
            index={i: rows[i] for i in range(3)}
        )

        mean_df = res_df.copy()
        for exp in rows:
            mean_df.loc[exp,:] = mean_df.loc[exp,:].apply(lambda x: float(x[:5]) if not x == 0 else 0)

        std_df = res_df.copy()
        for exp in rows:
            std_df.loc[exp,:] = std_df.loc[exp,:].apply(lambda x: float(x[-4:]) if not x == 0 else 0)

        x = np.arange(mean_df.shape[1])+1

        self.data["mean_df"] = mean_df
        self.data["std_df"] = std_df
        self.data["datasets"] = datasets
        self.data["rows"] = rows
        self.data["x"] = x

    def _get_plot_dims(self):
        self.plot_dims = (8,10)
    
    def plot(self):
        self._set_style()

        fig = plt.figure(figsize=self.plot_dims)

        for i, dataset in enumerate(self.data["mean_df"].columns):

            max_std = self.data["std_df"].loc[:,dataset].max()

            ax = fig.add_subplot(6,1,i+1)

            for j, exp in enumerate(self.data["mean_df"].index):
                ax.plot(
                    [0,4],
                    [self.data["mean_df"].values[0,i]]*2,
                    '--',
                    color=self.colors[1],
                    linewidth=1
                )
                ax.plot(
                    [j+1]*2,
                    [
                        self.data["mean_df"].values[j,i]-self.data["std_df"].values[j,i],
                        self.data["mean_df"].values[j,i]+self.data["std_df"].values[j,i]
                    ],
                    '-',
                    color='black',
                    label=exp,
                    linewidth=1
                )
                ax.plot(
                    j+1,
                    self.data["mean_df"].values[j,i],
                    'o',
                    color='black',
                    label=exp
                )

            ax.set_xticks(self.data["x"])
            ax.set_xticklabels(self.data["mean_df"].index if i == 5 else "")
            ax.set_xlim(0.5,3.5)

            start, end = ax.get_ylim()
            ticks = np.arange(np.round((start - max_std)/10)*10,np.round((end + max_std)/10)*10,10).tolist()
            if np.max(ticks) != 100:
                ticks.append(np.max(ticks)+10)
            ax.set_yticks(ticks)
            ax.yaxis.set_label_position("right")
            ax.set_ylabel('\\textbf{'+self.data["datasets"][i]+'}',size=13, rotation=270, labelpad=20)
            
        handles = [mpl.lines.Line2D([0],[0],color=self.colors[1], linestyle='--')]
        labels = ['Du et al. performance']
        fig.legend(
            handles, 
            labels, 
            loc = 'lower center', 
            fontsize=16, 
            bbox_to_anchor = (0,-0.1,1,1),
            bbox_transform = plt.gcf().transFigure, 
            borderaxespad=0.1, 
            ncol=1
        )

        fig.text(0.51, -0.02, 'Experiment', ha='center', va='center', rotation='horizontal', size=20)
        fig.text(-0.02, 0.5, 'Classification Accuracy (\%)', ha='center', va='center', rotation='vertical', size=20)
        plt.tight_layout(rect=(0.06, 0.06, 1, 1))
        plt.subplots_adjust(hspace=0)

        plt.tight_layout()
            
        plt.savefig(f'{self.config.reporting_path}/figures/exp_a_evaluation_2.png', bbox_inches='tight')


if __name__ == "__main__":
    exp_a_evaluation_2().plot()