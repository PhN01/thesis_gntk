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

class data_stats(Plot):

    def __init__(self):
        super(data_stats, self).__init__()

        self.config = cfg.Config()
        
        self.data = {}
        self._get_data()
        self._get_plot_dims()


    def _get_data(self):

        stats_df = pd.read_csv(f"{self.config.reporting_path}/data_statistics.csv", sep=",").drop(4, axis=0)

        self.data["stats_df"] = stats_df

    def _get_plot_dims(self):
        self.plot_dims = (10,5)
    
    def plot(self):
        self._set_style()

        fig = plt.figure(figsize=self.plot_dims)

        ax = fig.add_subplot(1,1,1)
        ax.plot(self.data["stats_df"]['avg_nodes'], self.data["stats_df"]['avg_edges'].values, 'o', color = 'black',)

        ax.text(
            self.data["stats_df"].loc[self.data["stats_df"].dataset=='COLLAB','avg_nodes'] -1, 
            self.data["stats_df"].loc[self.data["stats_df"].dataset=='COLLAB','avg_edges'] -50 , 
            'COLLAB', 
            horizontalalignment = 'right',
            fontdict={'size':16}
        )

        ax.set_xlabel('Average number of nodes')
        ax.set_ylabel('Average number of edges')
        ax.set_xlim(0,80)

        plt.tight_layout()
        plt.savefig(f'{self.config.reporting_path}/figures/data_stats.png', bbox_inches='tight')