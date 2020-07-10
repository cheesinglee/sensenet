import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()
kl = sensenet.importers.import_keras_layers()

from sensenet.layers.utils import initializer_map, activation_function

def get_shape_params(params):
    if 'kernel' in params:
        point_shape = depth_shape = np.array(params['kernel']).shape
    elif 'depth_kernel' in params:
        point_shape = np.array(params['point_kernel']).shape
        depth_shape = np.array(params['depth_kernel']).shape
    else:
        point_shape = depth_shape = None

    if depth_shape:
        kernel_dimensions = depth_shape[:2]

        if point_shape and len(point_shape) > 3:
            nfilters = point_shape[3]
        else:
            nfilters = None
    else:
        nfilters = int(params['number_of_filters'])
        kernel_dimensions = tuple(params['kernel_dimensions'])

    return nfilters, kernel_dimensions

def conv_2d(params):
    imap = initializer_map(params)
    nfilters, kernel_dimensions = get_shape_params(params)

    return kl.Conv2D(filters=nfilters,
                     kernel_size=kernel_dimensions,
                     dtype=tf.float32,
                     strides=params['strides'],
                     padding=params['padding'],
                     use_bias=params.get('bias', None) is not None,
                     activation=activation_function(params),
                     kernel_initializer=imap['kernel'],
                     kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                     bias_initializer=imap['bias'])


def separable_conv_2d(params):
    imap = initializer_map(params)
    nfilters, kernel_dimensions = get_shape_params(params)

    return kl.SeparableConv2D(filters=nfilters,
                              kernel_size=kernel_dimensions,
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
    imap = initializer_map(params)
    nfilters, kernel_dimensions = get_shape_params(params)

    return kl.DepthwiseConv2D(kernel_size=kernel_dimensions,
                              strides=params['strides'],
                              dtype=tf.float32,
                              padding=params['padding'],
                              use_bias=params.get('bias', None) is not None,
                              depth_multiplier=params['depth_multiplier'],
                              activation=activation_function(params),
                              depthwise_initializer=imap['kernel'],
                              bias_initializer=imap['bias'])

CONVOLUTIONAL_LAYERS = {
    "convolution_2d": conv_2d,
    "depthwise_convolution_2d": depthwise_conv_2d,
    "separable_convolution_2d": separable_conv_2d
}
