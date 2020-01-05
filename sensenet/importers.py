"""Namespace just for importing other namespaces to avoid spamming of
various messages on import.

"""
import sys
import os
import logging
import warnings

logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.WARNING)

from contextlib import contextmanager

@contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='.*as a synonym of type.*')
    import tensorflow

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='.*binary incompatibility.*')
    import numpy
    import scipy
    import sklearn

    import scipy.stats
    import scipy.special

    import sklearn.tree
    import sklearn.ensemble

with suppress_stderr():
    import keras

def import_tensorflow():
    return tensorflow

def import_numpy():
    return numpy

def import_keras():
    return keras

def import_scipy_stats():
    return scipy.stats

def import_scipy_special():
    return scipy.special

def import_sklearn_tree():
    return sklearn.tree

def import_sklearn_ensemble():
    return sklearn.ensemble
