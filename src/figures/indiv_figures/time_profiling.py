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


class time_profiling(Plot):

    def __init__(self):
        super(time_profiling, self).__init__()

        self.config = cfg.Config()
        
        self.data = {}
        self._get_data()
        self._get_plot_dims()

    def _get_data(self):

        with open(f"{self.config.exp_path}/GNTK/profiling/results_10samples.txt", "r") as f:
            profiles = json.load(f)
        stats = pd.read_csv("./reporting/data_statistics.csv", sep=",")

        datasets = [
            'NCI1',
            'MUTAG',
            'PTC',
            'PROTEINS',
            'IMDBMULTI',
            'IMDBBINARY',
            'COLLAB'
        ]
        datasets_alt = [
            'NCI1',
            'MUTAG',
            'PTC\_MR',
            'PROTEINS',
            'IMDB-MULTI',
            'IMDB-BINARY',
            'COLLAB'
        ]

        x = []
        y = {"s1": [], "s2": [], "s3": []}
        for dataset in datasets:
            x.append(float(stats.loc[stats.dataset==dataset,'avg_edges']))
            y["s1"].append(profiles["n10_L1_R3"][dataset]["Kronecker Product Calculation"])
            y["s2"].append(profiles["n10_L3_R3"][dataset]["Kronecker Product Calculation"])
            y["s3"].append(profiles["n10_L10_R3"][dataset]["Kronecker Product Calculation"])

        text_config = {
            "s1": {
                'NCI1': (50, -3.5, 'left'),
                'MUTAG': (50, 1, 'left'),
                'PTC': (50, -2.5, 'left'),
                'PROTEINS': (50, 3, 'left'),
                'IMDBMULTI': (50, 0, 'left'),
                'IMDBBINARY': (50, 0, 'left'),
                'COLLAB': (-30, -2, 'right')
            },
            "s2": {
                'NCI1': (50, -1, 'left'),
                'MUTAG': (50, -1, 'left'),
                'PTC': (50, 2, 'left'),
                'PROTEINS': (50, 0, 'left'),
                'IMDBMULTI': (50, 0, 'left'),
                'IMDBBINARY': (50, 2.5, 'left'),
                'COLLAB': (-30, -2, 'right')
            },
            "s3": {
                'NCI1': (50, -4, 'left'),
                'MUTAG': (50, -2.7, 'left'),
                'PTC': (50, 0, 'left'),
                'PROTEINS': (50, 0, 'left'),
                'IMDBMULTI': (50, 0, 'left'),
                'IMDBBINARY': (50, 0, 'left'),
                'COLLAB': (-30, -2, 'right')
            }
        }

        self.data["x"] = x
        self.data["y"] = y
        self.data["datasets"] = datasets
        self.data["datasets_alt"] = datasets_alt
        self.data["text_config"] = text_config

    def _get_plot_dims(self):
        self.plot_dims = (21,7)
    
    def plot(self):
        self._set_style()

        fig = plt.figure(figsize=self.plot_dims)
        axes = {}
        axes["s1"] = plt.subplot(131)
        axes["s2"] = plt.subplot(132, sharex = axes["s1"], sharey = axes["s1"])
        axes["s3"] = plt.subplot(133, sharex = axes["s1"], sharey = axes["s1"])

        for subplot, axis in axes.items():
            axis.plot(self.data["x"], self.data["y"][subplot], '.', color='black')
            for i, dataset in enumerate(self.data["datasets"]):
                axis.text(
                    self.data["x"][i] + self.data["text_config"][subplot][dataset][0],
                    self.data["y"][subplot][i] + self.data["text_config"][subplot][dataset][1],
                    self.data["datasets_alt"][i],
                    horizontalalignment = self.data["text_config"][subplot][dataset][2],
                    fontdict={'size':16}
                )
            axis.set_yticklabels(np.arange(10,101,10))
            if subplot == "s1":
                axis.set_ylim(15,101)
                axis.set_title('\\textbf{1 \\textsc{Block} Layers}', pad=15)
            elif subplot == "s2":
                axis.set_title('\\textbf{3 \\textsc{Block} Layers}', pad=15)
            elif subplot == "s3":
                axis.set_title('\\textbf{10 \\textsc{Block} Layers}', pad=15)

        # overall x text
        fig.text(0.5, 0.0, 'Dataset average number of edges', ha='center', va='center', size=20)
        fig.text(0.01, 0.5, '\% of CPU time spent on Kronecker Product computation', ha='center', va='center', rotation='vertical', size=20)

        plt.tight_layout(rect=(0.02, 0.03, 1, 1))
            
        plt.savefig(f'{self.config.reporting_path}/figures/time_profiling.png', bbox_inches='tight')


if __name__=="__main__":

    time_profiling().plot()