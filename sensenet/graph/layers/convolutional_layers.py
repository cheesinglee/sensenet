import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()
kl = sensenet.importers.import_keras_layers()

from sensenet.graph.layers.utils import initializer_map, activation_function

def conv_2d(params):
    kernel = np.array(params['kernel'])
    imap = initializer_map(params)

    return kl.Conv2D(filters=kernel.shape[3],
                     kernel_size=kernel.shape[:2],
                     dtype=tf.float32,
                     strides=params['strides'],
                     padding=params['padding'],
                     use_bias=params.get('bias', None) is not None,
                     activation=activation_function(params),
                     kernel_initializer=imap['kernel'],
                     bias_initializer=imap['bias'])


def separable_conv_2d(params):
    depth_shape = np.array(params['depth_kernel']).shape
    point_shape = np.array(params['point_kernel']).shape
    imap = initializer_map(params)

    return kl.SeparableConv2D(filters=point_shape[3],
                              kernel_size=depth_shape[:2],
                              dtype=tf.float32,
                              strides=params['strides'],
                              padding=params['padding'],
                              use_bias=params.get('bias', None) is not None,
                              depth_multiplier=params['depth_multiplier'],
                              activation=activation_function(params),
                              depthwise_initializer=imap['depth_kernel'],
                              pointwise_initializer=imap['point_kernel'],
                              bias_initializer=imap['bias'])

def depthwise_conv_2d(params):
    kernel = np.array(params['kernel'])
    imap = initializer_map(params)

    return kl.DepthwiseConv2D(kernel_size=kernel.shape[:2],
                              strides=params['strides'],
                              dtype=tf.float32,
                              padding=params['padding'],
                              use_bias=params.get('bias', None) is not None,
                              depth_multiplier=params['depth_multiplier'],
                              activation=activation_function(params),
                              depthwise_initializer=imap['kernel'],
                              bias_initializer=None)

CONVOLUTIONAL_LAYERS = {
    "convolution_2d": conv_2d,
    "depthwise_convolution_2d": depthwise_conv_2d,
    "separable_convolution_2d": separable_conv_2d
}
