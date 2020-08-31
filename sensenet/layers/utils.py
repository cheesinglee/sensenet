import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()
kl = sensenet.importers.import_keras_layers()

from sensenet.constants import LEAKY_RELU_ALPHA

ACTIVATORS = {
    'leaky_relu': lambda x: tf.nn.leaky_relu(x, alpha=LEAKY_RELU_ALPHA),
    'mish': lambda x: x * tf.math.tanh(tf.math.softplus(x)),
    'relu6': tf.nn.relu6,
    'swish': lambda x: x * tf.sigmoid(x)
}

# The names of the trainable parameters in any layer
WEIGHT_INITIALIZERS = {
    'weights': 'glorot_uniform',
    'offset': 'zeros',
    'gamma': 'ones',
    'scale': 'ones',
    'beta': 'zeros',
    'mean': 'zeros',
    'variance': 'ones',
    'stdev': 'ones',
    'kernel': 'glorot_uniform',
    'bias': 'zeros',
    'depth_kernel': 'glorot_uniform',
    'point_kernel': 'glorot_uniform'
}

INITIALIZERS = {
    'glorot_uniform': tf.initializers.glorot_uniform,
    'glorot_normal': tf.initializers.glorot_normal
}

def log_summary(x, msg):
    def summary_function(x):
        summary = [tf.shape(x), tf.reduce_mean(x), tf.math.reduce_std(x)]
        return tf.compat.v1.Print(x, summary, summarize=1024, message=msg)

    return kl.Lambda(summary_function)(x)

def variable(value, is_training, datatype=tf.float32):
    return tf.Variable(initial_value=value,
                       trainable=is_training,
                       dtype=datatype)

def constant(value, datatype=tf.float32):
    return tf.constant(value, dtype=datatype)

def transpose(amatrix):
    arr = np.array(amatrix)
    return np.transpose(arr).tolist()

def shape(tensor):
    return np.array(tensor.get_shape().as_list(), dtype=np.float32)

def activation_function(params):
    afn = params.get('activation_function', None)

    if afn in [None, 'linear', 'identity']:
        return 'linear'
    elif afn in ACTIVATORS:
        return ACTIVATORS[afn]
    else:
        return params['activation_function']

def initializer_map(params):
    imap = {}

    for k in WEIGHT_INITIALIZERS:
        weights = params.get(k, None)

        if isinstance(weights, str):
            if weights in ['zeros', 'ones']:
                imap[k] = weights
            else:
                random_seed = params.get('seed', 0)
                assert random_seed is not None
                imap[k] = INITIALIZERS[weights](seed=random_seed)
        elif weights is not None:
            weight_array = np.array(weights, dtype=np.float32)
            imap[k] = tf.constant_initializer(weight_array)
        else:
            imap[k] = 'zeros'

    return imap

def make_sequence(layers_params, creation_functions):
    layers = []

    if layers_params:
        for params in layers_params:
            layer_fn = creation_functions[params['type']]
            layers.append(layer_fn(params))

    return layers

def propagate(layers, inputs):
    next_inputs = inputs

    if len(layers) > 0:
        for layer in layers:
            next_inputs = layer(next_inputs)

    return next_inputs
