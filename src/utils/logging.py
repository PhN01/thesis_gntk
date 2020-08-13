import os
import colorlog
import logging
import datetime
import time
import math

import sys

sys.path.append(".")

from src.utils import utils


class Logger:
    __logger = None
    __has_filehandler = False
    __has_streamhandler = False

    class __Logger:
        def __init__(self, log_path):

            # the logger to use
            self.__logger = colorlog.getLogger()
            self.__has_filehandler = False
            self.__has_streamhandler = False

            # the format for logging to console
            console_formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(message)s",
                log_colors={
                    "DEBUG": "bold_cyan",
                    "INFO": "bold_green",
                    "WARNING": "bold_yellow",
                    "ERROR": "bold_red",
                    "CRITICAL": "white,bg_red",
                },
            )
            self.__console_handler = logging.StreamHandler()
            self.__console_handler.setFormatter(
                logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
            )
            self.__console_handler.setFormatter(console_formatter)
            self.__logger.addHandler(self.__console_handler)
            self.__has_streamhandler = True

            if not log_path is None:
                utils.make_dirs_checked(log_path)

                ts = datetime.datetime.now().strftime("%Y%m%d-T%H:%M:%S")
                path_split = sys.argv[0].split("/")
                src_idx = path_split.index("src")
                log_file = f"{'.'.join(path_split[src_idx:])}-{ts}.log"
                file_handler = logging.FileHandler(
                    f"{log_path}/{log_file}", mode="a", encoding=None, delay=False
                )
                file_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s [%(threadName)s] [%(levelname)s] %(message)s"
                    )
                )
                self.__logger.addHandler(file_handler)
                self.__has_filehandler = True

            # matplotlib adds annoying debug logs as soon as it is imported -> disable logging for matplotlib
            mpl_logger = logging.getLogger("matplotlib")
            mpl_logger.setLevel(logging.WARNING)

            # the logging level to consider
            self.__logger.setLevel(logging.DEBUG)

        def add_filehandler(self, log_path):
            if not self.__has_filehandler:
                utils.make_dirs_checked(log_path)

                ts = datetime.datetime.now().strftime("%Y%m%d-T%H:%M:%S")
                path_split = sys.argv[0].split("/")
                src_idx = path_split.index("src")
                log_file = f"{'.'.join(path_split[src_idx:])}-{ts}.log"
                file_handler = logging.FileHandler(
                    f"{log_path}/{log_file}", mode="a", encoding=None, delay=False
                )
                file_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s [%(threadName)s] [%(levelname)s] %(message)s"
                    )
                )
                self.__logger.addHandler(file_handler)
                self.__has_filehandler = True

        def set_level(self, level):
            self.__logger.setLevel(level)

        def debug(self, msg, end="\n"):
            self.__console_handler.terminator = end
            self.__logger.debug(msg)

        def info(self, msg, end="\n"):
            self.__console_handler.terminator = end
            self.__logger.info(msg)

        def warning(self, msg, end="\n"):
            self.__console_handler.terminator = end
            self.__logger.warning(msg)

        def error(self, msg, end="\n"):
            self.__console_handler.terminator = end
            self.__logger.error(msg)

        def critical(self, msg, end="\n"):
            self.__console_handler.terminator = end
            self.__logger.critical(msg)

    @staticmethod
    def get_logger(log_path=None):
        if Logger.__logger is None:
            Logger.__logger = Logger.__Logger(log_path=log_path)
        elif (Logger.__has_filehandler == False) & (not log_path is None):
            # Logger.__logger = None
            # Logger.__logger = Logger.__Logger(log_path=log_path)
            Logger.__logger.add_filehandler(log_path)
        return Logger.__logger


def get_logger(log_path=None):
    return Logger.get_logger(log_path)