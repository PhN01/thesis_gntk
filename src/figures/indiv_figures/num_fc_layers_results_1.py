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


class num_fc_layers_results_1(Plot):

    def __init__(self):
        super(num_fc_layers_results_1, self).__init__()

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

        jitter = [-0.05,0.05]

        self.data["datasets"] = datasets
        self.data["datasets_alt"] = datasets_alt
        self.data["K_val_df"] = K_val_df
        self.data["jitter"] = jitter

    def _get_plot_dims(self):
        self.plot_dims = (20,6)
    
    def plot(self):
        self._set_style()

        fig = plt.figure(figsize=self.plot_dims)

        for i, dataset in enumerate(self.data["datasets"]):
            ax = fig.add_subplot(1,6,i+1)
            for k, R in enumerate([1, 2, 3]):
                for j, scale in enumerate(['uniform', 'degree']):
                    df = self.data["K_val_df"].loc[
                        (self.data["K_val_df"].dataset==dataset)&\
                        (self.data["K_val_df"].norm==False)&\
                        (self.data["K_val_df"].scale==scale)&\
                        (self.data["K_val_df"].R==R),:
                    ]
                    ax.plot(
                        [k+1+self.data["jitter"][j]]*df.shape[0],
                        df['acc_mean'],
                        '.',
                        color=self.colors[j],
                        label=f'scale = {scale}' if (i == 5) & (k == 0) else None,
                    )
            
            if i == 0:
                ax.set_ylabel('Average validation accuracy ([0,1])')
            
            # x axes
            ax.set_xticks([1,2,3])
            ax.set_xticklabels([1,2,3])
            ax.set_xlim(0.5,3.5)
            
            # y axes
            start, end = ax.get_ylim()
            ax.set_yticks(np.arange(np.floor(start/0.1)*0.1,np.ceil(end/0.1)*0.1+0.1,0.1))
            
            # title
            ax.set_title('\\textbf{'+self.data["datasets_alt"][i]+'}')

        # overall x text
        fig.text(0.5, -0.02, 'Number of Fully Connected Layers', ha='center', va='center', size=20)

        # legend
        handles = [
            mpl.patches.Patch(color=self.colors[0]),
            mpl.patches.Patch(color=self.colors[1]),
        ]
        labels = [
            'scale = uniform',
            'scale = degree'
        ]
        fig.legend(handles, labels, loc = 'lower center', fontsize=16, bbox_to_anchor = (0,-0.15,1,1),
                    bbox_transform = plt.gcf().transFigure, borderaxespad=0.1, ncol=2)

        # final adjmustments and plotting
        plt.subplots_adjust(right=0.90)
        plt.tight_layout()
        plt.gca().yaxis.set_major_formatter(self.fmtr)
            
        plt.savefig(f'{self.config.reporting_path}/figures/num_fc_layers_results_1.png', bbox_inches='tight')


if __name__=="__main__":
    num_fc_layers_results_1().plot()