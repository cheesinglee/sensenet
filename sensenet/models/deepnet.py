import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import IMAGE_PATH, NUMERIC
from sensenet.accessors import get_output_exposition
from sensenet.layers.utils import propagate
from sensenet.layers.tree import ForestPreprocessor
from sensenet.layers.construct import layer_sequence, tree_preprocessor
from sensenet.preprocess.preprocessor import Preprocessor

def instantiate_inputs(model):
    preprocessors = model['preprocess']

    ncols = len(preprocessors)
    nimages = [p['type'] for p in preprocessors].count(IMAGE_PATH)

    Input = tf.keras.Input

    return {
        'numeric': Input((ncols,), dtype=tf.float32, name='numeric'),
        'image': Input((nimages,), dtype=tf.string, name='image')
    }

def deepnet_model(model, extras):
    outex = get_output_exposition(model)
    preprocessor = Preprocessor(model, extras)
    trees = tree_preprocessor(model)

    if 'networks' in model:
        all_layer_sequences = [layer_sequence(net) for net in model['networks']]
        use_trees = [net['trees'] for net in model['networks']]
    else:
        all_layer_sequences = [layer_sequence(model)]
        use_trees = [trees is not None]

    raw_inputs = instantiate_inputs(model)
    inputs = preprocessor(raw_inputs)

    if any(use_trees):
        treeed_inputs = trees(inputs)

    all_predictions = []

    for lseq, tree_in in zip(all_layer_sequences, use_trees):
        if tree_in:
            preds = propagate(lseq, treeed_inputs)
        else:
            preds = propagate(lseq, inputs)

        if outex['type'] == NUMERIC:
            preds = preds * outex['stdev'] + outex['mean']

        all_predictions.append(preds)

    if len(all_predictions) > 1:
        summed = tf.add_n(all_predictions)
        predictions = summed / len(all_predictions)
    else:
        predictions = all_predictions[0]

    return tf.keras.Model(inputs=raw_inputs, outputs=predictions)


# class DeepnetModel(tf.keras.Model):
#     def __init__(self, model):
#         super(DeepnetModel, self).__init__()

#         self._output_exposition = get_output_exposition(model)
#         self._preprocessor = Preprocessor(model)
#         self._trees = tree_preprocessor(model)

#         self._ensemble = 'networks' in model

#         if self._ensemble:
#             self._layers = [layer_sequence(net) for net in model['networks']]
#             self._use_trees = [net['trees'] for net in model['networks']]
#         else:
#             self._layers = [layer_sequence(model)]
#             self._use_trees = [self._trees is not None]

#         self._treeify = any(self._use_trees)

#     def call(self, inputs):
#         all_predictions = []

#         if self._treeify:
#             tree_data = self._trees(inputs)
#         else:
#             tree_data = None

#         for layers, use_trees in zip(self._layers, self._use_trees):
#             if use_trees:
#                 preds = propagate(layers, treeed_inputs)
#             else:
#                 preds = propagate(layers, inputs)

#             if self._output_exposition['type'] == NUMERIC:
#                 mean = self._output_exposition['mean']
#                 stdev = self._output_exposition['stdev']

#                 preds = preds * stdev + mean

#             all_predictions.append(preds)

#         if self._ensemble:
#             summed = tf.add_n(all_predictions)
#             return summed / len(all_predictions)
#         else:
#             return all_predictions[0]


# def initialize_variables(model, extras):
#     preprocs = model['preprocess']
#     variables = {'raw_X': tf.placeholder(tf.float32, (None, len(preprocs)))}
#     img_count = [p['type'] for p in preprocs].count(IMAGE_PATH)

#     if img_count:
#         variables['image_paths'] = tf.placeholder(tf.string, (None, img_count))

#     if extras:
#         variables.update(extras)

#     return variables

# def create_preprocessor(model, variables):
#     output_variables = {}

#     locations, load_vars = create_loaders(model, variables)
#     output_variables.update(load_vars)

#     if model.get('image_network', None):
#         img_vars = image_preprocessor(model, variables)
#         output_variables.update(img_vars)

#     preprocessed = reorder_inputs(locations, output_variables)
#     output_variables['preprocessed_X'] = preprocessed

#     if model['trees']:
#         embedded_X = forest_preprocessor(model, output_variables)
#         output_variables['embedded_X'] = embedded_X

#     return output_variables

# def create_network(network, variables, output_exposition=None):
#     outex = output_exposition or get_output_exposition(network)

#     if 'trees' in network and network['trees']:
#         Xin = variables['embedded_X']
#     else:
#         Xin = variables['preprocessed_X']

#     _, outputs = make_layers(Xin, network['layers'], None)

#     if outex['type'] == NUMERIC:
#         mten = make_tensor(outex['mean'])
#         stdten = make_tensor(outex['stdev'])

#         return outputs * stdten + mten
#     else:
#         return outputs

# def create_classifier(model, extras):
#     variables = initialize_variables(model, extras)
#     variables.update(create_preprocessor(model, variables))

#     if 'networks' in model:
#         outex = get_output_exposition(model)
#         mod_preds = []

#         for network in model['networks']:
#             preds = create_network(network, variables, output_exposition=outex)
#             mod_preds.append(preds)

#         summed = tf.add_n(mod_preds)
#         netslen = make_tensor(len(mod_preds))

#         variables['predictions'] = summed / netslen
#     else:
#         variables['predictions'] = create_network(model, variables)

#     return variables
