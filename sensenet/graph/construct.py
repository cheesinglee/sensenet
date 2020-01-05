"""These functions are used to generate tensorflow layers from a
topology spcification given in a dictionary, and to extract learned
parameters from the layers to export the model.

"""

import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from sensenet.graph.layers.utils import ACTIVATORS, PATH_KEYS, NORMALIZING_LAYERS
from sensenet.graph.layers.utils import PREVIOUS_INPUT_LAYERS, is_tf_variable
from sensenet.graph.layers.core_layers import CORE_LAYERS
from sensenet.graph.layers.convolutional_layers import CONVOLUTIONAL_LAYERS

def block_layer(X, params, is_training, paths, type_str):
    layer = {'type': type_str}
    outpaths = []

    for path in paths:
        if params[path] is not None:
            players, output = make_layers(X, is_training, params[path])
            layer[path] = players
            outpaths.append(output)
        else:
            layer[path] = []
            outpaths.append(X)

    outputs = outpaths[0] + outpaths[1];

    if 'activation_function' in params:
        afn = params['activation_function']
        outputs = ACTIVATORS[afn](outputs)
    else:
        afn = 'identity'

    layer['activation_function'] = afn

    return layer, outputs

def dense_res_block(X, params, is_training):
    paths = ['dense_path', 'identity_path']
    type_str = 'dense_residual_block'

    return block_layer(X, params, is_training, paths, type_str)

def xception_block(X, params, is_training):
    paths = ['separable_convolution_path', 'single_convolution_path']
    type_str = 'xception_block'

    return block_layer(X, params, is_training, paths, type_str)

def resnet_block(X, params, is_training):
    paths = ['convolution_path', 'identity_path']
    type_str = 'resnet_block'

    return block_layer(X, params, is_training, paths, type_str)

def darknet_block(X, params, is_training):
    paths = ['convolution_path', 'identity_path']
    type_str = 'darknet_residual_block'

    return block_layer(X, params, is_training, paths, type_str)

def mobilev2_block(X, params, is_training):
    paths = ['convolution_path', 'identity_path']
    type_str = 'mobilenet_residual_block'

    return block_layer(X, params, is_training, paths, type_str)

def xception_block(X, params, is_training):
    paths = ['separable_convolution_path', 'single_convolution_path']
    type_str = 'xception_block'

    return block_layer(X, params, is_training, paths, type_str)

def yolo_output_branches(layer_outputs, params, is_training):
    net = params['base_image_network']
    branches = params['output_branches']

    layer = {
        'type': 'yolo_output_branches',
        'base_image_network': net,
    }

    outputs = []
    out_branches = []

    for i, branch in enumerate(branches):
        branch_input = layer_outputs[branch['input']]
        path = branch['convolution_path']
        players, output = make_layers(branch_input, is_training, path)

        out_branch = dict(branch)
        out_branch['convolution_path'] = players

        out_branches.append(out_branch)
        outputs.append(output)

    layer['output_branches'] = out_branches

    return layer, outputs

BLOCK_LAYERS = {
    'xception_block': xception_block,
    'dense_residual_block': dense_res_block,
    'resnet_block': resnet_block,
    'resnet18_block': resnet_block,
    'darknet_residual_block': darknet_block,
    'mobilenet_residual_block': mobilev2_block,
    'yolo_output_branches': yolo_output_branches
}

LAYER_FUNCTIONS = {}
LAYER_FUNCTIONS.update(CORE_LAYERS)
LAYER_FUNCTIONS.update(CONVOLUTIONAL_LAYERS)
LAYER_FUNCTIONS.update(BLOCK_LAYERS)

def make_all_outputs(X, is_training, layers_params, keep_prob):
    outlayers = []
    all_outputs = []

    inputs = outputs = X

    for lp in layers_params:
        layer_type = lp['type']
        layer_fn = LAYER_FUNCTIONS[layer_type]

        if layer_type in NORMALIZING_LAYERS:
            layer, outputs = layer_fn(inputs, lp, is_training)
        elif layer_type in PREVIOUS_INPUT_LAYERS:
            layer, outputs = layer_fn(inputs, lp, is_training, all_outputs)
        elif layer_type == 'dropout':
            layer, outputs = layer_fn(inputs, lp, keep_prob)
        else:
            layer, outputs = layer_fn(inputs, lp)

        outlayers.append(layer)
        all_outputs.append(outputs)

        inputs = outputs

    return outlayers, all_outputs

def make_layers(X, is_training, layers_params, keep_prob=None):
    layers, outputs = make_all_outputs(X, is_training, layers_params, keep_prob)

    if layers:
        return layers, outputs[-1]
    else:
        return [], X

def place_values(out_layers, var_layers):
    for olayer, vlayer in zip(out_layers, var_layers):
        for key in vlayer:
            if key in PATH_KEYS:
                place_values(olayer[key], vlayer[key])
            else:
                assert is_tf_variable(olayer[key])
                olayer[key] = vlayer[key].tolist()

def layers_from_graph(layers, tf_session):
    out_layers = []
    var_layers = []

    for layer in layers:
        if 'type' not in layer or layer['type'] != 'dropout':
            out_layer = dict(layer)
            var_layer = {}

            for key in layer:
                if key in PATH_KEYS:
                    ols, vls = layers_from_graph(layer[key], None)
                    out_layer[key] = ols
                    var_layer[key] = vls
                elif is_tf_variable(layer[key]):
                    out_layer[key] = layer[key]
                    var_layer[key] = layer[key]
                elif isinstance(layer[key], tuple):
                    out_layer[key] = list(layer[key])

            out_layers.append(out_layer)
            var_layers.append(var_layer)

    if tf_session is None:
        return out_layers, var_layers
    else:
        eval_layers = tf_session.run(var_layers)
        place_values(out_layers, eval_layers)

        return out_layers
