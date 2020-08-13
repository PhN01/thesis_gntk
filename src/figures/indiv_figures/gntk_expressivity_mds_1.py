import numpy as np
import json
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from sklearn.manifold import MDS
from sklearn.manifold import TSNE

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

class gntk_expressivity_mds_1(Plot):

    def __init__(self):
        super(gntk_expressivity_mds_1, self).__init__()

        self.config = cfg.Config()
        
        self.data = {}
        self._get_data()
        self._get_plot_dims()


    def _get_data(self):

        def mds_coordinates(preds_df, seed=42):
            X = preds_df.values
            M = squareform(pdist(X, metric='hamming'))
            mds = MDS(
                metric=True,
                dissimilarity='precomputed',
                random_state=seed
            )
            x_mds = mds.fit_transform(M)

            return x_mds

        data_path = os.path.join(self.config.reporting_path, "exp_b_predictions.json")
        
        with open(data_path, "r") as f:
            preds_raw = json.load(f)

        target_len = {dataset:len(preds_raw['GNTK'][dataset]) for dataset in preds_raw['GNTK'].keys()}

        preds = {}
        for k,v in preds_raw.items():
            kernel = k.replace("_","\_")
            preds[kernel] = v

        kernels = list(preds.keys())
        datasets = list(preds['GNTK'].keys())

        dataset_alt = [
            'IMDB-BINARY',
            'IMDB-MULTI',
            'MUTAG',
            'NCI1',
            'PROTEINS',
            'PTC\_MR'
        ]

        kernels_alt = {
            'GNTK': 'GNTK',
            'VH': 'VH',
            'EH': 'EH',
            'HGKWL\\_seed0': 'WL-HGK',
            'HGKSP\\_seed0': 'SP-HGK',
            'MLG': 'MLK',
            'MP': 'MP',
            'SP': 'SP',
            'WL': 'WL',
            'WLOA': 'WL-OA',
            'GIN': 'GIN'
        }

        preds_combined = {}
        for kernel in kernels:
            pred_list = []
            for dataset in datasets:
                if preds[kernel][dataset] == 'NA':
                    preds[kernel][dataset] = [np.nan] * target_len[dataset]
                pred_list += preds[kernel][dataset]
                preds_combined[kernel] = pred_list
        preds_df = pd.DataFrame.from_dict(preds_combined,orient='index')

        offset = {
            'GNTK': (0.01, 0),
            'VH': (0.01, 0),
            'EH': (0.01, 0), 
            'WL-HGK': (0.02, -0.02), 
            'SP-HGK': (0.01, 0),
            'MLK': (0.01, 0), 
            'MP': (0.01, 0), 
            'SP': (0.01, 0), 
            'WL': (0.01,0), 
            'WL-OA': (-0.05,0.02),
            'GIN': (0.01,0)
        }

        self.data["x_mds"] = mds_coordinates(preds_df)
        self.data["preds_df"] = preds_df
        self.data["dataset_alt"] = dataset_alt
        self.data["kernels_alt"] = kernels_alt
        self.data["offset"] = offset

    def _get_plot_dims(self):
        self.plot_dims = (10,8)
    
    def plot(self):
        self._set_style()

        fig = plt.figure(figsize=self.plot_dims)

        ax = fig.add_subplot(1,1,1)
        for i,kernel in enumerate(self.data["preds_df"].index):
            ax.plot(
                self.data["x_mds"][i,0],
                self.data["x_mds"][i,1], 
                'o',
                color=self.colors[0] if kernel != 'GNTK' else 'red'
            )
            ax.text(
                self.data["x_mds"][i,0]+self.data["offset"][self.data["kernels_alt"][kernel]][0], 
                self.data["x_mds"][i,1]+self.data["offset"][self.data["kernels_alt"][kernel]][1], 
                self.data["kernels_alt"][kernel], 
                fontdict={'size':20}
            )
            
        start, end = ax.get_xlim()
        ax.set_xlim(start, np.ceil(end/0.1)*0.1+0.1)

        start, end = ax.get_ylim()
        ax.set_ylim(start, np.ceil(end/0.1)*0.1+0.1)
            
        fig.tight_layout()
        plt.gca().yaxis.set_major_formatter(self.fmtr)

        plt.savefig(f'{self.config.reporting_path}/figures/prediction_embedding_1.png', bbox_inches='tight')


if __name__=="__main__":

    gntk_expressivity_mds_1().plot()