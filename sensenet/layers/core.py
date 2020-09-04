"""Specifications for the most basic sort of network layers from JSON
to a Tensorflow graph.

"""

import sensenet.importers
tf = sensenet.importers.import_tensorflow()
kl = sensenet.importers.import_keras_layers()

from sensenet.layers.dropblock import DropBlock2D
from sensenet.layers.utils import initializer_map, activation_function

def get_units(params):
    if isinstance(params['weights'], str):
        return int(params['number_of_nodes'])
    else:
        return len(params['weights'][0])

def dense(params):
    imap = initializer_map(params)

    return kl.Dense(get_units(params),
                    dtype=tf.float32,
                    activation=activation_function(params),
                    use_bias=True,
                    kernel_initializer=imap['weights'],
                    bias_initializer=imap['offset'])

def activation(params):
    return kl.Activation(activation_function(params))

def batchnorm(params):
    imap = initializer_map(params)
    return kl.BatchNormalization(dtype=tf.float32,
                                 epsilon=params.get('epsilon', 1e-3),
                                 beta_initializer=imap['beta'],
                                 gamma_initializer=imap['gamma'],
                                 moving_mean_initializer=imap['mean'],
                                 moving_variance_initializer=imap['variance'])

def dropout(params):
    dtype = params['dropout_type']
    rate = params['rate']

    if dtype == 'zero':
        return kl.Dropout(rate=rate, seed=42)
    elif dtype == 'alpha':
        return kl.AlphaDropout(rate=rate, seed=42)
    elif dtype == 'block':
        bsize = params['block_size']
        return DropBlock2D(rate=rate, block_size=bsize, seed=42)
    else:
        raise ValueError('"%s" is not a valid dropout type!' % dtype)

def flatten(params):
    return kl.Flatten()

def global_avg_pool_2d(params):
    return kl.GlobalAveragePooling2D()

def global_max_pool_2d(params):
    return kl.GlobalMaxPool2D()

def max_pool_2d(params):
    pool_size = params['pool_size']
    strides = params['strides']
    padding = params['padding']

    return kl.MaxPool2D(pool_size, strides, padding)

def avg_pool_2d(params):
    pool_size = params['pool_size']
    strides = params['strides']
    padding = params['padding']

    return kl.AveragePooling2D(pool_size, strides, padding)

def padding_2d(params):
    padding = tuple([tuple([int(p) for p in ps]) for ps in params['padding']])
    return kl.ZeroPadding2D(padding)

def upsampling_2d(params):
    if params.get('method', None) == 'bilinear':
        return kl.UpSampling2D(params['size'], interpolation='bilinear')
    else:
        return kl.UpSampling2D(params['size'])

def concatenate(params):
    return kl.Concatenate()

def split_channels(params):
    n = params['number_of_splits']
    i = params['group_index']

    return kl.Lambda(lambda x: tf.split(x, num_or_size_splits=n, axis=-1)[i])

CORE_LAYERS = {
    'activation': activation,
    'average_pool_2d': avg_pool_2d,
    'batch_normalization': batchnorm,
    'concatenate': concatenate,
    'dense': dense,
    'dropout': dropout,
    'flatten': flatten,
    'global_average_pool_2d': global_avg_pool_2d,
    'global_max_pool_2d': global_max_pool_2d,
    'max_pool_2d': max_pool_2d,
    'padding_2d': padding_2d,
    'split_channels': split_channels,
    'upsampling_2d': upsampling_2d
}
