from abc import ABC
from abc import abstractmethod

import matplotlib as mpl
import matplotlib.pyplot as plt

class Plot(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def _get_data(self):
        pass

    @abstractmethod
    def _get_plot_dims(self):
        pass

    def _set_style(self, style=None):
        if style is None:
            mpl.style.use('seaborn')
        else:
            mpl.style.use(style)
        mpl.rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
        mpl.rc('text', usetex=True)
        mpl.rc('xtick', labelsize=20)
        mpl.rc('ytick', labelsize=20)
        mpl.rc('axes', labelsize=20, labelpad=10)
        mpl.rc('legend', fontsize=20)
        mpl.rc('axes', titlesize=20)

        self.colors = list(mpl.colors.TABLEAU_COLORS.keys())
        self.colors += self.colors
        self.markers = ['o','s','D']

        self.fmtr = mpl.ticker.StrMethodFormatter('{x:.1f}')

    @abstractmethod
    def plot(self):
        pass