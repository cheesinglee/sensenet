import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.graph.layers.utils import is_tf_variable, make_tensor

def bias_add(conv, bias):
    if bias is not None:
        return tf.nn.bias_add(conv, bias)
    else:
        return conv

def conv_2d(X, params, is_training):
    strides = params['strides']
    padding = params['padding']

    layer = {"type": "convolution_2d", "padding": padding, "strides": strides}

    for key in ["kernel", "bias"]:
        if not is_tf_variable(params[key]) and params[key] is not None:
            layer[key] = make_tensor(params[key], is_training)
        else:
            layer[key] = params[key]

    w = layer['kernel']
    stride_4d = [1] + list(strides) + [1]

    conv = tf.nn.conv2d(X, w, stride_4d, padding=padding.upper())
    outputs = bias_add(conv, layer['bias'])

    return layer, outputs

def separable_conv_2d(X, params, is_training):
    strides = params['strides']
    padding = params['padding']
    depth_multiplier = params["depth_multiplier"]

    layer = {
        "type": "separable_convolution_2d",
        "padding": padding,
        "strides": strides,
        "depth_multiplier": depth_multiplier
    }

    for key in ["depth_kernel", "point_kernel", "bias"]:
        if not is_tf_variable(params[key]) and params[key] is not None:
            layer[key] = make_tensor(params[key], is_training)
        else:
            layer[key] = params[key]

    dw = layer['depth_kernel']
    pw = layer['point_kernel']
    stride_4d = [1] + list(strides) + [1]

    conv = tf.nn.separable_conv2d(X, dw, pw, stride_4d, padding=padding.upper())
    outputs = bias_add(conv, layer['bias'])

    return layer, outputs

def depthwise_conv_2d(X, params, is_training):
    strides = params['strides']
    padding = params['padding']
    depth_multiplier = params["depth_multiplier"]

    layer = {
        "type": "depthwise_convolution_2d",
        "padding": padding,
        "strides": strides,
        "depth_multiplier": depth_multiplier
    }

    for key in ["kernel", "bias"]:
        if not is_tf_variable(params[key]) and params[key] is not None:
            layer[key] = make_tensor(params[key], is_training)
        else:
            layer[key] = params[key]

    w = layer['kernel']
    stride_4d = [1] + list(strides) + [1]

    conv = tf.nn.depthwise_conv2d(X, w, stride_4d, padding=padding.upper())
    outputs = bias_add(conv, layer['bias'])

    return layer, outputs

CONVOLUTIONAL_LAYERS = {
    "convolution_2d": conv_2d,
    "depthwise_convolution_2d": depthwise_conv_2d,
    "separable_convolution_2d": separable_conv_2d
}
