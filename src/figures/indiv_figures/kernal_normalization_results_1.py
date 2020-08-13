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


class kernel_normalization_results_1(Plot):

    def __init__(self):
        super(kernel_normalization_results_1, self).__init__()

        self.config = cfg.Config()
        
        self.data = {}
        self._get_data()
        self._get_plot_dims()

    def _get_data(self):

        datasets = ['IMDBBINARY','IMDBMULTI','MUTAG','NCI1','PROTEINS','PTC']
        datasets_alt = [
            'IMDB-BINARY',
            'IMDB-MULTI',
            'MUTAG',
            'NCI1',
            'PROTEINS',
            'PTC\_MR'
        ]

        K_val_df = pd.DataFrame()

        for dataset in datasets:
            df = pd.read_csv(f'{self.config.eval_path}/GNTK/b.1/{dataset}/K_validation_df.csv', sep=',')
            df['dataset'] = dataset
            K_val_df = pd.concat([K_val_df, df], axis=0)

        jitter = [-0.1,0.1]

        x = np.arange(1,15)

        self.data["datasets"] = datasets
        self.data["datasets_alt"] = datasets_alt
        self.data["K_val_df"] = K_val_df
        self.data["jitter"] = jitter
        self.data["x"] = x

    def _get_plot_dims(self):
        self.plot_dims = (20,22)
    
    def plot(self):
        self._set_style()

        fig = plt.figure(figsize=self.plot_dims)

        for i, dataset in enumerate(self.data["datasets"]):
            ax = fig.add_subplot(3,2,i+1)
            c = 0
            for k, norm in enumerate([False,True]):
                if norm == False:
                    for j, scale in enumerate(['uniform','degree']):
                        df = self.data["K_val_df"].loc[
                            (self.data["K_val_df"].dataset==dataset)&\
                            (self.data["K_val_df"].norm==norm)&\
                            (self.data["K_val_df"].scale==scale),:
                        ]
                        ax.plot(
                            df['L'],
                            df['acc_mean'],
                            '.',
                            color=self.colors[c]
                        )
                        c+=1
                elif norm == True:
                    for j, scale in enumerate(['uniform','degree']):
                        df = self.data["K_val_df"].loc[
                            (self.data["K_val_df"].dataset==dataset)&\
                            (self.data["K_val_df"].norm==norm)&\
                            (self.data["K_val_df"].scale==scale),:
                        ]
                        ax.plot(
                            df['L']+self.data["jitter"][j],
                            df['acc_mean'],
                            '.',
                            color=self.colors[c],
                            linewidth=1
                        )
                        c+=1
            
            ax.set_xlabel('Number of \\textsc{Block} layers')
            ax.set_xticks(np.arange(1,15))
            ax.set_xticklabels(np.arange(1,15))
            ax.set_xlim(0.5,14.5)

            start, end = ax.get_ylim()
            ax.set_yticks(np.arange(np.floor(start/0.1)*0.1,np.ceil(end/0.1)*0.1+0.1,0.1))
            ax.set_ylabel('Average validation accuracy ([0,1])')
            ax.set_title('\\textbf{'+self.data["datasets_alt"][i]+'}')
        #     ax.set_ylim(0,5.05)
            
        handles = [
            mpl.patches.Patch(color=self.colors[0]),
            mpl.patches.Patch(color=self.colors[1]),
            mpl.patches.Patch(color=self.colors[2]),
            mpl.patches.Patch(color=self.colors[3]),
        ]
        labels = [
            'norm = False, scale = uniform',
            'norm = False, scale = degree',
            'norm = True, scale = uniform',
            'norm = True, scale = degree'
        ]
        fig.legend(handles, labels, loc = 'lower center', fontsize=16, bbox_to_anchor = (0,0,1,1),
                    bbox_transform = plt.gcf().transFigure, borderaxespad=0.1, ncol=2)

        plt.tight_layout(rect=(0.03, 0.03, 1, 1))
        plt.subplots_adjust(hspace=0.3)
        plt.gca().yaxis.set_major_formatter(self.fmtr)
            
        plt.savefig(f'{self.config.reporting_path}/figures/kernel_normalization_results_1.png', bbox_inches='tight')

if __name__=="__main__":
    kernel_normalization_results_1().plot()