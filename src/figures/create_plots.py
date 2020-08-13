from inspect import getmembers, isclass

import sys
sys.path.append(".")

from src.utils import utils
from src.utils import logging
from src import figures

logger = logging.get_logger()

if __name__=="__main__":

    plot_list = [o for o in getmembers(figures.indiv_figures) if isclass(o[1])]

    for i, (plot_name, plot_class) in enumerate(plot_list):

        # try:
        plot_class().plot()
        logger.info(f"Plot {i} -- {plot_name}")
        # except:
            # logger.warning(f"Plot {i} -- Creating plot {plot_name} failed.")