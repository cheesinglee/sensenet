def extract_padding_2d(layer):
    config = layer.get_config()

    return {
        "type": "padding_2d",
        "padding": config['padding'],
        "value": 0
    }

def extract_max_pool_2d(layer):
    config = layer.get_config()

    return {
        "type": "max_pool_2d",
        "pool_size": config['pool_size'],
        "strides": config['strides'],
        "padding": config['padding']
    }

def extract_avg_pool_2d(layer):
    config = layer.get_config()

    return {
        "type": "average_pool_2d",
        "pool_size": config['pool_size'],
        "strides": config['strides'],
        "padding": config['padding']
    }

def extract_global_avg_pool_2d(layer):
    return {
        "type": "global_average_pool_2d"
    }

def extract_global_max_pool_2d(layer):
    return {
        "type":"global_max_pool_2d"
    }

def extract_separable_conv_2d(layer):
    config = layer.get_config()

    if config['use_bias']:
        depth_kernel, point_kernel, bias = layer.get_weights()

        return {
            "type": "separable_convolution_2d",
            "strides": config['strides'],
            "padding": config['padding'],
            "depth_kernel": depth_kernel.tolist(),
            "point_kernel": point_kernel.tolist(),
            "depth_multiplier": config['depth_multiplier'],
            "bias": bias.tolist()
        }
    else:
        depth_kernel, point_kernel = layer.get_weights()
        return {
            "type": "separable_convolution_2d",
            "strides": config['strides'],
            "padding": config['padding'],
            "depth_kernel": depth_kernel.tolist(),
            "point_kernel": point_kernel.tolist(),
            "depth_multiplier": config['depth_multiplier'],
            "bias": None
        }

def extract_depth_conv_2d(layer):
    config = layer.get_config()

    if config['use_bias']:
        kernel, bias = layer.get_weights()
        return {
            "type": "depthwise_convolution_2d",
            "strides": config['strides'],
            "padding": config['padding'],
            "kernel": kernel.tolist(),
            "depth_multiplier": config['depth_multiplier'],
            "bias": bias.tolist()
        }
    else:
        kernel = layer.get_weights()[0]
        return {
            "type": "depthwise_convolution_2d",
            "strides": config['strides'],
            "padding": config['padding'],
            "kernel": kernel.tolist(),
            "depth_multiplier": config['depth_multiplier'],
            "bias": None
        }

def extract_conv_2d(layer):
    config = layer.get_config()

    if config['use_bias']:
        kernel, bias = layer.get_weights()
        return {
            "type": "convolution_2d",
            "strides": config['strides'],
            "padding": config['padding'],
            "kernel": kernel.tolist(),
            "bias": bias.tolist()
        }
    else:
        kernel = layer.get_weights()[0]
        return {
            "type": "convolution_2d",
            "strides": config['strides'],
            "padding": config['padding'],
            "kernel": kernel.tolist(),
            "bias": None
        }

def extract_batchnorm(layer):
    gamma, beta, mean, variance = layer.get_weights()
    epsilon = layer.get_config()['epsilon']

    return {
        "type": "batch_normalization",
        "gamma": gamma.tolist(),
        "beta": beta.tolist(),
        "mean": mean.tolist(),
        "variance": variance.tolist(),
        "epsilon": epsilon
    }


def extract_activation(layer):
    config = layer.get_config()
    return {"type": "activation", "activation_function": config['activation']}

def extract_leaky_relu(layer):
    return {"type": "activation", "activation_function": "leaky_relu"}

def extract_flatten(layer):
    return {"type": "flatten"}

def extract_dense(layer):
    config = layer.get_config()
    weights, offset = layer.get_weights()

    return {
        "type": "dense",
        "activation_function": config.get("activation", None),
        "weights": weights.tolist(),
        "offset": offset.tolist(),
    }

def extract_reshape(layer):
    config = layer.get_config()

    return {
        "type": "reshape",
        "target_shape": config["target_shape"]
    }

def extract_dropout(layer):
    config = layer.get_config()

    return {
        "type": "dropout",
        "dropout_type": "zero",
        "rate": config["rate"]
    }

def extract_upsampling_2d(layer):
    config = layer.get_config()

    return {
        "type": "upsampling_2d",
        "size": config["size"]
    }

def extract_concatenate(layer):
    return {"type": "concatenate"}

EXTRACTORS = {
    "Activation": extract_activation,
    "LeakyReLU": extract_leaky_relu,
    "BatchNormalization": extract_batchnorm,
    "Conv2D": extract_conv_2d,
    "Concatenate": extract_concatenate,
    "UpSampling2D": extract_upsampling_2d,
    "DepthwiseConv2D": extract_depth_conv_2d,
    "SeparableConv2D": extract_separable_conv_2d,
    "ZeroPadding2D": extract_padding_2d,
    "AveragePooling2D": extract_avg_pool_2d,
    "GlobalAveragePooling2D": extract_global_avg_pool_2d,
    "GlobalMaxPooling2D": extract_global_max_pool_2d,
    "MaxPooling2D": extract_max_pool_2d,
    "Flatten": extract_flatten,
    "Dense": extract_dense,
    "Reshape": extract_reshape,
    "Dropout": extract_dropout
}

def kType(layer):
    return str(layer.__class__.__name__)

def extract(layer, get_name=False):
    layer = EXTRACTORS[kType(layer)](layer)

    if get_name:
        layer[name] = layer.name

    return layer
