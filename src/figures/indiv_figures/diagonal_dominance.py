import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import pickle
import graphkernels as gk

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
from src.data import data_loaders
from src.data import graph_utils


class diagonal_dominance(Plot):

    def __init__(self):
        super(diagonal_dominance, self).__init__()

        self.config = cfg.Config()
        
        self.data = {}
        self._get_data()
        self._get_plot_dims()

    def _get_data(self):

        with open(f'{self.config.matrix_path_duetal}/PTC/gram_PTC_L1_R3_scaledegree_jk0/gram.pkl','rb') as f:
            gntk_l1 = pickle.load(f)
        with open(f'{self.config.matrix_path_duetal}/PTC/gram_PTC_L5_R3_scaledegree_jk0/gram.pkl','rb') as f:
            gntk_l5 = pickle.load(f)
        with open(f'{self.config.matrix_path_duetal}/PTC/gram_PTC_L14_R3_scaledegree_jk0/gram.pkl','rb') as f:
            gntk_l14 = pickle.load(f)

        graphs, labels = data_loaders.load_graphs_tudortmund("PTC")

        igraph_list = graph_utils.conv_graph_nx2ig(graphs)

        wl_h1 = gk.CalculateWLKernel(igraph_list, par=1)
        wl_h5 = gk.CalculateWLKernel(igraph_list, par=5)
        wl_h14 = gk.CalculateWLKernel(igraph_list, par=14)

        np.random.seed(42)
        samples = np.random.choice(np.arange(len(graphs)), 70, replace=False).tolist()

        self.data["gntk"] = {"l1": gntk_l1, "l5": gntk_l5, "l14": gntk_l14}
        self.data["wl"] = {"h1": wl_h1, "h5": wl_h5, "h14": wl_h14}
        self.data["samples"] = samples

    def _get_plot_dims(self):
        self.plot_dims = (15,12)
    
    def plot(self):
        self._set_style(style="default")

        fig = plt.figure(figsize=self.plot_dims)

        fig = plt.figure(figsize=(15,12))

        ax = fig.add_subplot(2,3,1)
        im = ax.matshow(self.data["wl"]["h1"][self.data["samples"],:][:,self.data["samples"]])
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title('\\boldmath$h = 1$')

        ax = fig.add_subplot(2,3,2)
        im = ax.matshow(self.data["wl"]["h5"][self.data["samples"],:][:,self.data["samples"]])
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title('\\boldmath$h = 5$')

        ax = fig.add_subplot(2,3,3)
        im = ax.matshow(self.data["wl"]["h14"][self.data["samples"],:][:,self.data["samples"]])
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title('\\boldmath$h = 14$')

        ax = fig.add_subplot(2,3,4)
        im = ax.matshow(self.data["gntk"]["l1"][self.data["samples"],:][:,self.data["samples"]])
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title('\\boldmath$L = 1$')

        ax = fig.add_subplot(2,3,5)
        im = ax.matshow(self.data["gntk"]["l5"][self.data["samples"],:][:,self.data["samples"]])
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title('\\boldmath$L = 5$')

        ax = fig.add_subplot(2,3,6)
        im = ax.matshow(self.data["gntk"]["l14"][self.data["samples"],:][:,self.data["samples"]])
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title('\\boldmath$L = 14$')

        fig.text(0.0, 0.75, '\\textbf{WL Subtree Kernel}', ha='center', va='center', rotation='vertical', size=20)
        fig.text(0.0, 0.25, '\\textbf{GNTK}', ha='center', va='center', rotation='vertical', size=20)
        plt.subplots_adjust(hspace=0.4)

        plt.tight_layout()
            
        plt.savefig(f'{self.config.reporting_path}/figures/diagonal_dominance.png', bbox_inches='tight')


if __name__ == "__main__":

    diagonal_dominance().plot()