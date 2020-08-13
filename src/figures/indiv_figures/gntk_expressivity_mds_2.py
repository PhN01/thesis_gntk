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

class gntk_expressivity_mds_2(Plot):

    def __init__(self):
        super(gntk_expressivity_mds_2, self).__init__()

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

        x_mds_bydatasets = {}
        preds_df_bydataset = {}
        for dataset in datasets:
            dataset_preds = {}
            for kernel in kernels:
                dataset_preds[kernel] = preds[kernel][dataset]
            dataset_preds_df = pd.DataFrame.from_dict(dataset_preds,orient='index')
            if dataset == 'PROTEINS':
                dataset_preds_df = dataset_preds_df.drop('SP', axis=0)

            x_mds_bydatasets[dataset] = mds_coordinates(dataset_preds_df)
            preds_df_bydataset[dataset] = dataset_preds_df

        offset = {
            'IMDBBINARY':
            {
                'GNTK': (0, 0.02),
                'VH': (0, 0.02),
                'EH': (0, 0.02), 
                'WL-HGK': (0.015, -0.01), 
                'SP-HGK': (0.015, 0),
                'MLK': (0, 0.02), 
                'MP': (0, 0.02), 
                'SP': (0.015, -0.01), 
                'WL': (0, 0.02), 
                'WL-OA': (0.015, -0.01),
                'GIN': (0,0.02)
            },
            'IMDBMULTI':
            {
                'GNTK': (0, 0.02),
                'VH': (0, 0.02),
                'EH': (0, 0.02), 
                'WL-HGK': (0.015, -0.01), 
                'SP-HGK': (-0.05, 0.02),
                'MLK': (0, 0.02), 
                'MP': (0, 0.02), 
                'SP': (0.015, -0.01), 
                'WL': (0.015, 0.01), 
                'WL-OA': (-0.07, 0.015),
                'GIN': (0,0.02)
            },
            'MUTAG':
            {
                'GNTK': (0, 0.02),
                'VH': (0, 0.02),
                'EH': (0, 0.02), 
                'WL-HGK': (0, 0.02), 
                'SP-HGK': (0, 0.02),
                'MLK': (0, 0.02), 
                'MP': (0, 0.02), 
                'SP': (0, 0.02), 
                'WL': (0.01, -0.01), 
                'WL-OA': (0, 0.02),
                'GIN': (0,0.02)
            },
            'NCI1':
            {
                'GNTK': (0, 0.02),
                'VH': (0, 0.02),
                'EH': (0, 0.02), 
                'WL-HGK': (0, 0.02), 
                'SP-HGK': (0, 0.02),
                'MLK': (0, 0.02), 
                'MP': (0, 0.02), 
                'SP': (0, 0.02), 
                'WL': (0, 0.02), 
                'WL-OA': (-0.01, -0.05),
                'GIN': (0,0.02)
            },
            'PROTEINS':
            {
                'GNTK': (0, 0.01),
                'VH': (0, 0.01),
                'EH': (0, 0.01), 
                'WL-HGK': (0, 0.01), 
                'SP-HGK': (-0.01, 0.01),
                'MLK': (0, 0.01), 
                'MP': (0, 0.01), 
                'SP': (0, 0.01), 
                'WL': (0, 0.01), 
                'WL-OA': (0, 0.01),
                'GIN': (0.02, 0)
            },
            'PTC':
            {
                'GNTK': (0, 0.02),
                'VH': (0, 0.02),
                'EH': (0, 0.02), 
                'WL-HGK': (0, 0.02), 
                'SP-HGK': (0, 0.02),
                'MLK': (0, 0.02), 
                'MP': (0, 0.02), 
                'SP': (0, 0.02), 
                'WL': (0.02, 0), 
                'WL-OA': (0, 0.02),
                'GIN': (0,0.02)
            },
        }

        self.data["x_mds_bydataset"] = x_mds_bydatasets
        self.data["preds_df_bydataset"] = preds_df_bydataset
        self.data["datasets"] = datasets
        self.data["dataset_alt"] = dataset_alt
        self.data["kernels_alt"] = kernels_alt
        self.data["offset"] = offset

    def _get_plot_dims(self):
        self.plot_dims = (20,22)
    
    def plot(self):
        self._set_style()

        fig = plt.figure(figsize=self.plot_dims)

        for i, dataset in enumerate(self.data["datasets"]):

            
            y_max = self.data["x_mds_bydataset"][dataset][:,1].max()
            y_lim = y_max - self.data["x_mds_bydataset"][dataset][:,1].min()
            
            ax = fig.add_subplot(3,2,i+1)
            for j,kernel in enumerate(self.data["preds_df_bydataset"][dataset].index):
                ax.plot(
                    self.data["x_mds_bydataset"][dataset][j,0],
                    self.data["x_mds_bydataset"][dataset][j,1], 
                    'o',
                    color=self.colors[0] if kernel != 'GNTK' else 'red'
                )
                ax.text(
                    self.data["x_mds_bydataset"][dataset][j,0]+self.data["offset"][dataset][self.data["kernels_alt"][kernel]][0], 
                    self.data["x_mds_bydataset"][dataset][j,1]+self.data["offset"][dataset][self.data["kernels_alt"][kernel]][1], 
                    self.data["kernels_alt"][kernel], 
                    fontdict={'size':20}
                )
                
            ax.set_title('\\textbf{'+self.data["dataset_alt"][i]+'}')

            start, end = ax.get_xlim()
            ax.set_xlim(start, np.ceil(end/0.1)*0.1+0.1)

            start, end = ax.get_ylim()
            ax.set_ylim(start, np.ceil(end/0.1)*0.1+0.1)
            
            plt.locator_params(axis='x', nbins=10)
            
        plt.tight_layout(rect=(0.03, 0.03, 1, 1))
        plt.subplots_adjust(hspace=0.3)
        plt.gca().yaxis.set_major_formatter(self.fmtr)

        plt.savefig(f'{self.config.reporting_path}/figures/prediction_embedding_2.png', bbox_inches='tight')


if __name__=="__main__":

    gntk_expressivity_mds_2().plot()