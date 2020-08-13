import os
import colorlog
import logging
import datetime
import time
import math

import PIL.Image as Image


def make_dirs_checked(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def split_path(path):
    directory, filename = os.path.split(path)
    basename, extension = os.path.splitext(filename)
    return (directory, basename, extension)


def make_path(path_tup):
    if isinstance(path_tup, tuple):
        if len(path_tup) == 2:
            dir, file = path_tup
            path = os.path.join(dir, file)
        elif len(path_tup) == 3:
            dir, file, ext = path_tup
            path = os.path.join(dir, file + ext)
    else:
        path = path_tup

    return path

