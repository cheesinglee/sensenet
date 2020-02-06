import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()
kl = sensenet.importers.import_keras_layers()

from sensenet.constants import LEAKY_RELU_ALPHA

ACTIVATORS = {
    'relu6': tf.nn.relu6,
    'swish': lambda x: x * tf.sigmoid(x),
    'leaky_relu': lambda x: tf.nn.leaky_relu(x, alpha=LEAKY_RELU_ALPHA)
}

# The names of the trainable parameters in any layer
WEIGHT_PARAMETERS = [
    'weights',
    'offset',
    'gamma',
    'beta',
    'mean',
    'variance',
    'kernel',
    'bias',
    'depth_kernel',
    'point_kernel'
]

def log_summary(x, msg):
    def summary_function(x):
        summary = [tf.shape(x), tf.reduce_mean(x), tf.math.reduce_std(x)]
        return tf.compat.v1.Print(x, summary, summarize=1024, message=msg)

    return kl.Lambda(summary_function)(x)

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

    for k in WEIGHT_PARAMETERS:
        wts = params.get(k, None)

        if wts is not None:
            imap[k] = tf.constant_initializer(np.array(wts, dtype=np.float32))
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

def variable(value, is_training, datatype=tf.float32):
    return tf.Variable(initial_value=value,
                       trainable=is_training,
                       dtype=datatype)

def constant(value, datatype=tf.float32):
    return tf.constant(value, dtype=datatype)

def transpose(amatrix):
    arr = np.array(amatrix)
    return np.transpose(arr).tolist()
