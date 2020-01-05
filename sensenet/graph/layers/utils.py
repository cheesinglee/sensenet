import sensenet.importers
tf = sensenet.importers.import_tensorflow()
np = sensenet.importers.import_numpy()

from sensenet.constants import LEAKY_RELU_ALPHA

TF_VAR_TYPE = type(tf.Variable(0))

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
    'linear': tf.identity
}

# These are keys that map to arrays of layers
PATH_KEYS = ['convolution_path',
             'dense_path',
             'identity_path',
             'single_convolution_path',
             'seperable_convolution_path',
             'output_branches']

# These layers to batch normalization that needs to be turned off at
# prediction time
NORMALIZING_LAYERS = ['batch_normalization',
                      'resnet_block',
                      'dense_residual_block',
                      'xception_block',
                      'resnet18_block',
                      'mobilenet_residual_block',
                      'darknet_residual_block']

# These layers use inputs more than just those immediately preceeding
# them
PREVIOUS_INPUT_LAYERS = ['concatenate', 'yolo_output_branches']

ANCHORS = {
    'yolov3': np.array([[10, 13],
                        [16, 30],
                        [33, 23],
                        [30, 61],
                        [62, 45],
                        [59, 119],
                        [116 ,90],
                        [156 ,198],
                        [373 ,326]]),
    'tinyyolov3': np.array([[10, 14],
                            [23, 27],
                            [37, 58],
                            [81, 82],
                            [135, 169],
                            [344, 319]])
}

MASKS = {
    'yolov3': [[6,7,8], [3,4,5], [0,1,2]],
    'tinyyolov3': [[3,4,5], [0,1,2]]
}

OUTPUT_NODES = {
    'yolov3': [60, 80, 100],
    'tinyyolov3': [29, 34]
}

OUTPUT_CHANNELS = {
    'yolov3': [512, 256, 128],
    'tinyyolov3': [256, 384]
}

INTERMEDIATE_FILTER_SIZES = {
    'yolov3': [1024, 512, 256],
    'tinyyolov3': [512, 256]
}

def is_tf_variable(var):
    return type(var) == TF_VAR_TYPE
