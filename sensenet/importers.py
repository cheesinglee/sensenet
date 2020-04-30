"""Namespace just for importing other namespaces to avoid spamming of
various messages on import.

"""
import logging
import warnings

logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*binary incompatibility.*')

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='.*binary incompatibility.*')
    import numpy

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='.*as a synonym of type.*')
    warnings.filterwarnings('ignore', message='.*binary incompatibility.*')
    import tensorflow
    import tensorflow.keras.layers

def import_tensorflow():
    return tensorflow

def import_keras_layers():
    return tensorflow.keras.layers

def import_numpy():
    return numpy
