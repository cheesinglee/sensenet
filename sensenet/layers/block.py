"""These functions are used to generate tensorflow layers from a
topology spcification given in a dictionary, and to extract learned
parameters from the layers to export the model.

"""

import sensenet.importers
tf = sensenet.importers.import_tensorflow()
kl = sensenet.importers.import_keras_layers()

from sensenet.layers.utils import activation_function, make_sequence, propagate
from sensenet.layers.core import CORE_LAYERS
from sensenet.layers.convolutional import CONVOLUTIONAL_LAYERS

SIMPLE_LAYERS = {}
SIMPLE_LAYERS.update(CORE_LAYERS)
SIMPLE_LAYERS.update(CONVOLUTIONAL_LAYERS)

class BlockLayer(tf.keras.layers.Layer):
    def __init__(self, params, path_names, block_type):
        super(BlockLayer, self).__init__()

        self._block_name = block_type
        self._path_names = path_names
        self._paths = []

        for name in path_names:
            self._paths.append(make_sequence(params[name], SIMPLE_LAYERS))

        self._activator = kl.Activation(activation_function(params))

    def call(self, inputs):
        outputs = [propagate(p, inputs) for p in self._paths]
        activations = tf.add_n(outputs)

        return self._activator(activations)

def dense_res_block(params):
    paths = ['dense_path', 'identity_path']
    type_str = 'dense_residual_block'

    return BlockLayer(params, paths, type_str)

def xception_block(params):
    paths = ['separable_convolution_path', 'single_convolution_path']
    type_str = 'xception_block'

    return BlockLayer(params, paths, type_str)

def resnet_block(params):
    paths = ['convolution_path', 'identity_path']
    type_str = 'resnet_block'

    return BlockLayer(params, paths, type_str)

def darknet_block(params):
    paths = ['convolution_path', 'identity_path']
    type_str = 'darknet_residual_block'

    return BlockLayer(params, paths, type_str)

def mobilev2_block(params):
    paths = ['convolution_path', 'identity_path']
    type_str = 'mobilenet_residual_block'

    return BlockLayer(params, paths, type_str)

# def xception_block(params):
#     paths = ['separable_convolution_path', 'single_convolution_path']
#     type_str = 'xception_block'

#     return BlockLayer(params, paths, type_str)

BLOCKS = {
    'xception_block': xception_block,
    'dense_residual_block': dense_res_block,
    'resnet_block': resnet_block,
    'resnet18_block': resnet_block,
    'darknet_residual_block': darknet_block,
    'mobilenet_residual_block': mobilev2_block
}

# def make_all_outputs(X, layers_params, is_training, keep_prob):
#     outlayers = []
#     all_outputs = []

#     inputs = outputs = X
#     use_next = True

#     for i, lp in enumerate(layers_params):
#         if use_next:
#             layer_type = lp.get('type', 'legacy')
#             layer_fn = LAYER_FUNCTIONS[layer_type]

#             if i < len(layers_params) - 1:
#                 residuals = layers_params[i + 1].get('residuals', False)
#             else:
#                 residuals = False

#             if layer_type in PREVIOUS_INPUT_LAYERS:
#                 layer, outputs = layer_fn(inputs, lp, is_training, all_outputs)
#             elif layer_type == 'legacy':
#                 if residuals:
#                     params = [lp, layers_params[i + 1]]
#                     layer, outputs = convert_legacy(inputs, params, is_training)
#                     use_next = False
#                 else:
#                     layer, outputs = layer_fn(inputs, lp, is_training)

#             elif layer_type == 'dropout':
#                 layer, outputs = layer_fn(inputs, lp, keep_prob)
#             else:
#                 layer, outputs = layer_fn(inputs, lp, is_training)

#             outlayers.append(layer)
#             all_outputs.append(outputs)

#             inputs = outputs
#         else:
#             use_next = True

#     return outlayers, all_outputs

# def make_layers(X, layers_params, is_training, keep_prob=None):
#     layers, outputs = make_all_outputs(X, layers_params, is_training, keep_prob)

#     if layers:
#         return layers, outputs[-1]
#     else:
#         return [], X

# def place_values(out_layers, var_layers):
#     for olayer, vlayer in zip(out_layers, var_layers):
#         for key in vlayer:
#             if key in PATH_KEYS:
#                 place_values(olayer[key], vlayer[key])
#             else:
#                 assert is_tf_variable(olayer[key])
#                 olayer[key] = vlayer[key].tolist()

# def layers_from_graph(layers, tf_session):
#     out_layers = []
#     var_layers = []

#     for layer in layers:
#         if 'type' not in layer or layer['type'] != 'dropout':
#             out_layer = dict(layer)
#             var_layer = {}

#             for key in layer:
#                 if key in PATH_KEYS:
#                     ols, vls = layers_from_graph(layer[key], None)
#                     out_layer[key] = ols
#                     var_layer[key] = vls
#                 elif is_tf_variable(layer[key]):
#                     out_layer[key] = layer[key]
#                     var_layer[key] = layer[key]
#                 elif isinstance(layer[key], tuple):
#                     out_layer[key] = list(layer[key])

#             out_layers.append(out_layer)
#             var_layers.append(var_layer)

#     if tf_session is None:
#         return out_layers, var_layers
#     else:
#         eval_layers = tf_session.run(var_layers)
#         place_values(out_layers, eval_layers)

#         return out_layers
