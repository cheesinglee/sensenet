#!/usr/bin/env python

import sys
import json

from sensenet.layers.utils import WEIGHT_INITIALIZERS
from sensenet.layers.core import get_units
from sensenet.layers.convolutional import get_shape_params, CONVOLUTIONAL_LAYERS
from sensenet.layers.construct import layer_sequence

from utils import read_regression, make_model

COPY_KEYS = ['type', 'activation_function']

PATH_KEYS = [
    'dense_path',
    'convolution_path',
    'separable_convolution_path',
    'identity_path'
]

def get_weight_shapes(params):
    if params['type'] == 'dense':
        return {'number_of_nodes': get_units(params)}
    elif params['type'] in CONVOLUTIONAL_LAYERS:
        nfilters, kdim = get_shape_params(params)
        return {'number_of_filters': nfilters, 'kernel_dimensions': kdim}
    else:
        return {}

def extract_layers(layers):
    outputs = []

    for i, params in enumerate(layers):
        output = get_weight_shapes(params)

        if output:
            output['seed'] = i

        for key in params:
            if key in COPY_KEYS:
                output[key] = params[key]
            elif key in PATH_KEYS:
                output[key] = extract_layers(params[key])
            elif key in WEIGHT_INITIALIZERS:
                output[key] = WEIGHT_INITIALIZERS[key]
            else:
                raise ValueError('What is "%s"?' % str(key))

        outputs.append(output)

    return outputs

def add_dropouts(layers, drate):
    new_layers = [{'type': 'dropout', 'dropout_type': 'zero', 'rate': 0.8}]
    dense_types = ['dense', 'dense_residual_block']

    for i, layer in enumerate(layers):
        if i < len(layers) - 1 and layer['type'] in dense_types:

            if layer['activation_function'] == 'selu':
                dtype = 'alpha'
            else:
                dtype = 'zero'

            dlayer = {'type': 'dropout', 'dropout_type': dtype, 'rate': drate}

            new_layers.append(layer)
            new_layers.append(dlayer)
        else:
            new_layers.append(layer)

    return new_layers

def extract_regression_parameters(reg_file):
    test_artifact = read_regression(reg_file, root='.')
    all_options = []

    for i, test in enumerate(test_artifact):
        layers = extract_layers(test['model']['layers'])
        options = test['model']['training_options']

        drate = options.get('dropout_rate', 0.0)
        bn = options.get('batch_normalization', False)

        if drate and not bn:
            layers = add_dropouts(layers, drate)

        for key in ['randomize', 'activation_functions', 'layer_sizes',
                    'activation_function', 'learn_residuals',
                    'batch_normalizaiton', 'dropout_rate']:
            options.pop(key, None)

        make_model(layer_sequence({'layers': layers}), 6)

        options['layers'] = layers
        options['seed'] = i

        all_options.append(options)

    return all_options

def main():
    reg_file = sys.argv[1]
    out_file = sys.argv[2]
    params = extract_regression_parameters(reg_file)

    with open(out_file, 'w') as fout:
        json.dump(params, fout)

if __name__ == '__main__':
    main()
