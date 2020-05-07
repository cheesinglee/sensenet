from sensenet.layers.utils import WEIGHT_INITIALIZERS
from sensenet.layers.core import get_units
from sensenet.layers.convolutional import get_shape_params, CONVOLUTIONAL_LAYERS

COPY_KEYS = [
    'type',
    'activation_function',
    'padding',
    'strides',
    'size',
    'pool_size',
    'value',
    'depth_multiplier',
    'epsilon',
    'inputs',
    'input'
]

PATH_KEYS = [
    'dense_path',
    'convolution_path',
    'separable_convolution_path',
    'identity_path',
    'single_convolution_path',
    'output_branches'
]

IGNORE_KEYS = [
    'base_image_network',
    'anchors'
]

def get_weight_shapes(params):
    if 'type' not in params:
        return {}
    elif params['type'] == 'dense':
        return {'number_of_nodes': get_units(params)}
    elif params['type'] in CONVOLUTIONAL_LAYERS:
        nfilters, kdim = get_shape_params(params)
        return {'number_of_filters': nfilters, 'kernel_dimensions': kdim}
    else:
        return {}

def extract_layers(layers):
    outputs = []

    if layers:
        for i, params in enumerate(layers):
            output = get_weight_shapes(params)

            if output:
                output['seed'] = i

            if 'input' in params and 'convolution_path' in params:
                output['type'] = 'yolo_output_branch'

            for key in params:
                if key in COPY_KEYS:
                    output[key] = params[key]
                elif key in PATH_KEYS:
                    output[key] = extract_layers(params[key])
                elif key in WEIGHT_INITIALIZERS:
                    if params[key] is None:
                        output[key] = None
                    else:
                        output[key] = WEIGHT_INITIALIZERS[key]
                elif key in IGNORE_KEYS:
                    pass
                else:
                    if 'type' in params:
                        raise ValueError('What is "%s: %s" in a "%s" layer?' %
                                         (key, str(params[key]), params['type']))
                    else:
                        raise ValueError(params.keys())

            outputs.append(output)

    return outputs
