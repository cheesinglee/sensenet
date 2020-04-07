#!/usr/bin/env python

import sys
import json

from sensenet.constants import ANCHORS, IMAGE_PATH, CATEGORICAL, BOUNDING_BOX
from sensenet.layers.utils import WEIGHT_INITIALIZERS
from sensenet.layers.core import get_units
from sensenet.layers.convolutional import get_shape_params, CONVOLUTIONAL_LAYERS
from sensenet.layers.construct import layer_sequence

from sensenet.pretrained import get_pretrained_layers, get_pretrained_readout
from sensenet.pretrained import get_pretrained_network, PRETRAINED_CNN_METADATA

from utils import read_regression, make_model, image_model

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

def extract_image_artifacts(network, out_file):
    img_net_spec = get_pretrained_network(network)
    name = img_net_spec['metadata']['base_image_network']

    print("Reading %s..." % name)
    conv_layers = get_pretrained_layers(img_net_spec)
    readout_layers = get_pretrained_readout(img_net_spec)

    if 'yolo' in name:
        with open('tests/cococlasses.json', 'r') as fin:
            imgclasses = json.load(fin)

        otype = BOUNDING_BOX
        nclasses = 80
        conv_layers = conv_layers + readout_layers
        conv_specs = extract_layers(conv_layers)
        readout_layers = None
        readout_specs = None
    else:
        with open('tests/imgnetclasses.json', 'r') as fin:
            cmap = json.load(fin)

        imgclasses = [cmap[str(i)] for i in range(1000)]
        otype = CATEGORICAL
        conv_specs = extract_layers(conv_layers)
        readout_specs = extract_layers(readout_layers)

    img_net_spec['layers'] = conv_layers
    img_net_spec['metadata']['mean_image'] = None

    if 'output_indices' in img_net_spec['metadata']:
        anchors = ANCHORS[img_net_spec['metadata']['base_image_network']]
        img_net_spec['metadata']['anchors'] = anchors

    full_network = {
        'output_exposition': {'type': otype, 'values': imgclasses},
        'layers': readout_layers,
        'trees': None,
        'image_network': img_net_spec,
        'preprocess': [{'type': IMAGE_PATH, 'index': 0}]
    }

    model = image_model(full_network)
    model.save_weights(out_file)

    # boxes, scores, classes = model.predict([['tests/data/images/pizza_people.jpg']])
    # print(boxes, scores, classes)

    full_network['image_network']['layers'] = conv_specs
    full_network['layers'] = readout_specs

    print("Recon...")
    remodel = image_model(full_network)
    remodel.load_weights(out_file)

    # boxes, scores, classes = remodel.predict([['tests/data/images/pizza_people.jpg']])
    # print(boxes, scores, classes)

    return full_network

def extract_regression_parameters(reg_file):
    test_artifact = read_zipped_json(reg_file)
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

        make_model(layer_sequence({'layers': layers}), (6,))

        options['layers'] = layers
        options['seed'] = i

        all_options.append(options)

    return all_options

def main():
    reg_file = sys.argv[1]
    out_file = sys.argv[2]

    if reg_file.endswith('.json'):
        params = extract_regression_parameters(reg_file)

        with open(out_file, 'w') as fout:
            json.dump(params, fout)
    elif reg_file == 'image_networks':
        outmeta = {}

        for netid in PRETRAINED_CNN_METADATA:
        # for netid in ['mobilenet']:
            print(netid)
            netmeta = get_pretrained_network(netid)
            out_file = netid + "_" + netmeta['metadata']['version'] + ".h5"

            newmeta = extract_image_artifacts(netid, out_file)
            outmeta[netid] = newmeta

        with open('sensenet_metadata.json', 'w') as fout:
            json.dump(outmeta, fout, sort_keys=True, indent=4)

if __name__ == '__main__':
    main()
