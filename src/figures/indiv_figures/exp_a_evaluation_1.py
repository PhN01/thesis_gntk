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


class exp_a_evaluation_1(Plot):

    def __init__(self):
        super(exp_a_evaluation_1, self).__init__()

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
        self.plot_dims = (15,6)
    
    def plot(self):
        self._set_style()

        fig = plt.figure(figsize=self.plot_dims)

        ax = fig.add_subplot(1,1,1)

        ax.plot(
            self.data["x"],
            self.data["mean_df"].values[0,:],
            '-',
            color='black',
            label='Du et al.'
        )
        ax.plot(
            self.data["x"],
            self.data["mean_df"].values[1,:],
            '--',
            color=self.colors[0],
            label='(a.1)'
        )
        ax.plot(
            self.data["x"],
            self.data["mean_df"].values[2,:],
            '--',
            color=self.colors[1],
            label='(a.2)'
        )
        ax.set_xticks(self.data["x"])
        ax.set_xticklabels(self.data["datasets"], size=16)
        ax.set_xlim(0.5,7.3)
        ax.set_ylabel('Accuracy (\%)', size=20, labelpad=10)
        ax.set_yticks([int(i*10) for i in np.arange(5,10, 0.5)])
        ax.set_yticklabels([int(i*10) for i in np.arange(5,10, 0.5)], size=20)

        plt.legend(loc='right', fontsize=16)

        plt.tight_layout()
            
        plt.savefig(f'{self.config.reporting_path}/figures/exp_a_evaluation_1.png', bbox_inches='tight')


if __name__ == "__main__":
    exp_a_evaluation_1().plot()