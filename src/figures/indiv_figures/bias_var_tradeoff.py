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

class bias_var_tradeoff(Plot):

    def __init__(self):
        super(bias_var_tradeoff, self).__init__()

        self.config = cfg.Config()
        
        self.data = {}
        self._get_data()
        self._get_plot_dims()


    def _get_data(self):

        x = np.linspace(2*np.pi,3*np.pi,num=30)
        x_pand = np.expand_dims(x, axis=1)
        y = np.sin(x) + np.random.normal(scale=0.2, size=30)

        degree = {
            0:0,
            1:2,
            2:20
        }

        self.data["x"] = x
        self.data["x_pand"] = x_pand
        self.data["y"] = y
        self.data["degree"] = degree

    def _get_plot_dims(self):
        self.plot_dims = (20,6)
    
    def plot(self):
        self._set_style()

        fig = plt.figure(figsize=self.plot_dims)

        for i in range(3):
            ax = fig.add_subplot(1,3,i+1)
            ax.plot(self.data["x"], self.data["y"], '.', color = 'black')

            lin = LinearRegression() 
            x_poly = PolynomialFeatures(degree = self.data["degree"][i]).fit_transform(self.data["x_pand"])
            lin.fit(x_poly, self.data["y"])

            ax.plot(self.data["x"], lin.predict(x_poly), '-', color=self.colors[1])
            ax.set_xlabel('x') 
            ax.set_ylabel('y')
            
            ax.set_xticks(np.arange(6,10,0.5))
            ax.set_yticks(np.arange(0,1.75,0.25))
            
            plt.gca().xaxis.set_major_formatter(self.fmtr)

        plt.tight_layout()
        plt.savefig(f'{self.config.reporting_path}/figures/bias-var-tradeoff.png', bbox_inches='tight')