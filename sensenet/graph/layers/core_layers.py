"""Specifications for the most basic sort of network layers from JSON
to a Tensorflow graph.

"""

import sensenet.importers
tf = sensenet.importers.import_tensorflow()
kl = sensenet.importers.import_keras_layers()

# from sensenet.graph.layers.utils import is_tf_variable, ACTIVATORS
# from sensenet.graph.layers.utils import make_tensor, transpose

WEIGHT_PARAMETERS = [
    'weights',
    'offset',
    'gamma',
    'beta',
    'mean',
    'variance'
]

def activation_function(params):
    afn = params.get('activation_function', None)

    if afn in [None, 'linear', 'identity']:
        return 'linear'
    elif afn == 'swish':
        return lambda x: x * tf.sigmoid(x)
    else:
        return params['activation_function']

def initializer_map(params):
    imap = {}

    for k in WEIGHT_PARAMETERS:
        if k in params and params[k] is not None:
            imap[k] = tf.constant_initializer(params[k])

    imap['activation_function'] = activation_function(params)

    return imap

def dense(params):
    imap = initializer_map(params)
    units = len(params['weights'][0])

    return kl.Dense(units,
                    activation=imap['activation_function'],
                    use_bias=True,
                    kernel_initializer=imap['weights'],
                    bias_initializer=imap['offset'])

def activation(params):
    imap = initializer_map(params)
    return kl.Activation(imap['activation_function'])

def batchnorm(params):
    imap = initializer_map(params)
    return kl.BatchNormalization(beta_initializer=imap['beta'],
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

def flatten(X, params, is_training):
    return kl.Flatten()

def global_avg_pool_2d(X, params, is_training):
    return kl.GlobalAveragePooling2D()

def global_max_pool_2d(X, params, is_training):
    return kl.GlobalMaxPool2D()

def max_pool_2d(X, params, is_training):
    pool_size = params['pool_size']
    strides = params['strides']
    padding = params['padding']

    return kl.MaxPool2D(pool_size, strides, padding)

def avg_pool_2d(X, params, is_training):
    pool_size = params['pool_size']
    strides = params['strides']
    padding = params['padding']

    return kl.AveragePooling2D(pool_size, strides, padding)

def padding_2d(params):
    padding = [[int(p) for p in ps] for ps in params['padding']]
    return kl.ZeroPadding2D(padding)

def upsampling_2d():
    return kl.UpSampling2D(params['size'])

CORE_LAYERS = {
    'dense': dense,
    'activation': activation,
    'batch_normalization': batchnorm,
    'dropout': dropout,
    'flatten': flatten,
    'average_pool_2d': avg_pool_2d,
    'max_pool_2d': max_pool_2d,
    'global_average_pool_2d': global_avg_pool_2d,
    'global_max_pool_2d': global_max_pool_2d,
    'padding_2d': padding_2d,
    'upsampling_2d': upsampling_2d
}
