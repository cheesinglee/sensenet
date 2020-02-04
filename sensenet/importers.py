"""Namespace just for importing other namespaces to avoid spamming of
various messages on import.

"""
import sys
import os
import logging
import warnings

logging.getLogger('tensorflow').setLevel(logging.ERROR)

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
    import tensorflow.keras.layers

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='.*binary incompatibility.*')
    import numpy

def import_tensorflow():
    return tensorflow

def import_keras_layers():
    return tensorflow.keras.layers

def import_numpy():
    return numpy
