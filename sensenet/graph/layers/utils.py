import sensenet.importers
tf = sensenet.importers.import_tensorflow()
np = sensenet.importers.import_numpy()

from sensenet.constants import LEAKY_RELU_ALPHA

TF_VAR_TYPE = type(tf.Variable(0))
TF_PLACEHOLDER_TYPE = type(tf.placeholder(tf.float32))

ACTIVATORS = {
    'tanh': tf.tanh,
    'sigmoid': tf.sigmoid,
    'softplus': tf.nn.softplus,
    'softmax': tf.nn.softmax,
    'relu': tf.nn.relu,
    'relu6': tf.nn.relu6,
    'swish': lambda x: x * tf.sigmoid(x),
    'selu': tf.nn.selu,
    'leaky_relu': lambda x: tf.nn.leaky_relu(x, alpha=LEAKY_RELU_ALPHA),
    'identity': tf.identity,
    'linear': tf.identity,
    None: tf.identity
}

# These are keys that map to arrays of layers
PATH_KEYS = ['convolution_path',
             'dense_path',
             'identity_path',
             'single_convolution_path',
             'seperable_convolution_path',
             'output_branches']

# These layers use inputs more than just those immediately preceeding
# them
PREVIOUS_INPUT_LAYERS = ['concatenate', 'yolo_output_branches']

def is_tf_variable(var):
    return type(var) in [TF_VAR_TYPE, TF_PLACEHOLDER_TYPE]

def make_tensor(value, is_training=False, ttype=tf.float32):
    if is_training is None or is_training is False:
        return tf.constant(value, dtype=ttype)
    else:
        assert is_tf_variable(is_training)
        return tf.Variable(value, dtype=ttype)

def transpose(amatrix):
    arr = np.array(amatrix)
    return np.transpose(arr).tolist()
