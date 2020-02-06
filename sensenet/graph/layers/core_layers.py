"""Specifications for the most basic sort of network layers from JSON
to a Tensorflow graph.

"""

import sensenet.importers
tf = sensenet.importers.import_tensorflow()
kl = sensenet.importers.import_keras_layers()

# from sensenet.graph.layers.utils import is_tf_variable, ACTIVATORS
# from sensenet.graph.layers.utils import make_tensor, transpose
from sensenet.graph.layers.utils import initializer_map, activation_function

def dense(params):
    imap = initializer_map(params)
    units = len(params['weights'][0])

    return kl.Dense(units,
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
                                 beta_initializer=imap['beta'],
                                 gamma_initializer=imap['gamma'],
                                 moving_mean_initializer=imap['mean'],
                                 moving_variance_initializer=imap['variance'])

def dropout(params):
    dtype = params['dropout_type']
    rate = params['dropout_rate']

    if dtype == 'zero':
        return kl.Dropout(rate, seed=42)
    elif dtype == 'alpha':
        return kl.AlphaDropout(rate, seed=42)
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
    padding = [[int(p) for p in ps] for ps in params['padding']]
    return kl.ZeroPadding2D(padding)

def upsampling_2d(params):
    return kl.UpSampling2D(params['size'])

def concatenate(params):
    return kl.Concatenate()

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
    'upsampling_2d': upsampling_2d
}
