"""Specifications for the most basic sort of network layers from JSON
to a Tensorflow graph.

"""

import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from sensenet.graph.layers.utils import is_tf_variable, ACTIVATORS

def dense(X, params):
    afn = params['activation_function']
    layer = {"type": "dense"}

    for key in ["weights", "offset"]:
        if not is_tf_variable(params[key]):
            layer[key] = tf.Variable(params[key], dtype=tf.float32)
        else:
            layer[key] = params[key]

    w = layer['weights']
    offset_var = layer['offset']

    outputs = tf.matmul(X, w) + offset_var

    if afn in [None, "linear", "identity"]:
        layer['activation_function'] = "identity"
    else:
        layer['activation_function'] = afn
        outputs = ACTIVATORS[afn](outputs)

    return layer, outputs

def activation(X, params):
    afn = params['activation_function']
    layer = {"type": "activation", "activation_function": afn}

    outputs = ACTIVATORS[afn](X)

    return layer, outputs

def batchnorm(X, params, is_training, decay=0.99, eps=1e-3):
    layer = {"type": "batch_normalization"}

    if 'epsilon' in set(params.keys()):
        eps = params['epsilon']
    for key in ['gamma', 'beta', 'mean', 'variance']:
        if not is_tf_variable(params[key]):
            layer[key] = tf.Variable(params[key], dtype=tf.float32)
        else:
            layer[key] = params[key]

    gamma = layer['gamma']
    beta = layer['beta']
    pop_mean = layer['mean']
    pop_var = layer['variance']

    def train():
        m, v = tf.nn.moments(X, list(range(len(X.get_shape()) - 1)))
        train_mean = tf.assign(pop_mean, pop_mean * decay + m * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + v * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(X, m, v, beta, gamma, eps)

    def test():
        return tf.nn.batch_normalization(X, pop_mean, pop_var, beta, gamma, eps)

    outputs = tf.cond(is_training, train, test)

    return layer, outputs

def dropout(X, params, keep_prob):
    dtype = params['dropout_type']
    layer = {"type": "dropout", "dropout_type": dtype}

    if dtype == "zero":
        dropout_rate = 1 - keep_prob
        outputs = tf.nn.dropout(X, rate=dropout_rate, seed=42)
    elif dtype == "alpha":
        outputs = tf.contrib.nn.alpha_dropout(X, keep_prob, seed=42)
    else:
        raise ValueError("'%s' is not a valid dropout type!" % dtype)

    return layer, outputs

def flatten(X, params):
    layer = {"type": "flatten"}

    shape = X.get_shape().as_list()
    dim = np.prod(shape[1:])
    outputs = tf.reshape(X, [-1, dim])

    return layer, outputs

def global_avg_pool_2d(X, params):
    layer = {"type": "global_average_pool_2d"}
    outputs = tf.reduce_mean(X, axis=[1, 2])

    return layer, outputs

def global_max_pool_2d(X, params):
    layer = {"type": "global_max_pool_2d"}
    outputs = tf.reduce_max(X, axis=[1, 2])

    return layer, outputs

def max_pool_2d(X, params):
    pool_size = params['pool_size']
    strides = params['strides']
    padding = params['padding']

    layer = {
        "type": "max_pool_2d",
        "pool_size": pool_size,
        "strides": strides,
        "padding": padding
    }

    pool_4d = [1] + list(pool_size) + [1]
    stride_4d = [1] + list(strides) + [1]

    outputs = tf.nn.max_pool(X, pool_4d, stride_4d, padding.upper())

    return layer, outputs

def avg_pool_2d(X, params):
    pool_size = params['pool_size']
    strides = params['strides']
    padding = params['padding']

    layer = {
        "type": "average_pool_2d",
        "pool_size": pool_size,
        "strides": strides,
        "padding": padding
    }

    pool_4d = [1] + list(pool_size) + [1]
    stride_4d = [1] + list(strides) + [1]

    outputs = tf.nn.avg_pool(X, pool_4d, stride_4d, padding.upper())

    return layer, outputs

def padding_2d(X, params):
    padding = [[int(p) for p in ps] for ps in params['padding']]

    layer = {"type": "padding_2d", "value": 0, "padding": padding}
    pad_spec = tf.constant([[0, 0]] + padding + [[0, 0]])

    outputs = tf.pad(X, pad_spec)

    return layer, outputs

def upsampling_2d(X, params):
    size = params['size']

    layer = {
        "type": "upsampling_2d",
        "size": size
    }

    out_size = X.get_shape()[1:3] * np.array(size)
    rmeth = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    outputs = tf.image.resize_images(X, out_size, method=rmeth)

    return layer, outputs

def concatenate(X, params, is_training, all_outputs):
    inputs = params['inputs']

    layer = {
        "type": "concatenate",
        "inputs": params['inputs']
    }

    Xs = [all_outputs[inp] for inp in inputs]
    outputs = tf.concat(Xs, -1)

    return layer, outputs

CORE_LAYERS = {
    "dense": dense,
    "activation": activation,
    "batch_normalization": batchnorm,
    "dropout": dropout,
    "flatten": flatten,
    "average_pool_2d": avg_pool_2d,
    "max_pool_2d": max_pool_2d,
    "global_average_pool_2d": global_avg_pool_2d,
    "global_max_pool_2d": global_max_pool_2d,
    "padding_2d": padding_2d,
    "upsampling_2d": upsampling_2d,
    "concatenate": concatenate
}
