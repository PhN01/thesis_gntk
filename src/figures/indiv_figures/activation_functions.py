import numpy as np
import json
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression 

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


class activation_functions(Plot):

    def __init__(self):
        super(activation_functions, self).__init__()

        self.config = cfg.Config()
        
        self.data = {}
        self._get_data()
        self._get_plot_dims()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def d_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def d_relu(self, x):
        return np.array([0 if i < 0 else 1 for i in x])

    def d_tanh(self, x):
        return 1 - np.tanh(x)**2

    def _get_data(self):

        x = np.linspace(-6,6,num=1001)        
        yticks = np.arange(-1,1.25,0.25).tolist()
        yticks.remove(0)

        self.data["x"] = x
        self.data["yticks"] = yticks

    def _get_plot_dims(self):
        self.plot_dims = (20,6)
    
    def plot(self):
        self._set_style()

        fig = plt.figure(figsize=self.plot_dims)

        ax = fig.add_subplot(1,3,1)
        ax.plot(self.data["x"], np.tanh(self.data["x"]), '-', color=self.colors[1], linewidth=2)
        ax.plot(self.data["x"], self.d_tanh(self.data["x"]), '--', color=self.colors[0], linewidth=2)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$\\sigma(x)$')
        ax.set_ylim(-1.1,1.1)
        ax.set_title('\\textbf{Tanh activation}', pad=9)

        ax = fig.add_subplot(1,3,2)
        ax.plot(self.data["x"], self.sigmoid(self.data["x"]), '-', color=self.colors[1], linewidth=2)
        ax.plot(self.data["x"], self.d_sigmoid(self.data["x"]), '--', color=self.colors[0], linewidth=2)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$\\sigma(x)$')
        ax.set_ylim(-1.1,1.1)
        ax.set_title('\\textbf{Sigmoid activation}', pad=9)

        ax = fig.add_subplot(1,3,3)
        ax.plot(self.data["x"], self.relu(self.data["x"]), '-', color=self.colors[1], linewidth=2)
        ax.plot(self.data["x"], self.d_relu(self.data["x"]), '--', color=self.colors[0], linewidth=2)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$\\sigma(x)$')
        ax.set_ylim(-1.1,1.1)
        ax.set_title('\\textbf{ReLU activation}', pad=9)
        ax.set_xlim(-1,1)


        handles = [
            mpl.lines.Line2D([0],[0],color=self.colors[1], linestyle='--'),
            mpl.lines.Line2D([0],[0],color=self.colors[0], linestyle='--')
        ]
        labels = [
            '$\\sigma(x)$',
            '$\\frac{\\partial}{\\partial x}\\sigma(x)$'
        ]
        fig.legend(handles, labels, loc = 'lower center', fontsize=20, bbox_to_anchor = (0,-0.1,1,1),
                    bbox_transform = plt.gcf().transFigure, borderaxespad=0.1, ncol=2)

        plt.tight_layout()
            
        plt.savefig(f'{self.config.reporting_path}/figures/activation_functions.png', bbox_inches='tight')