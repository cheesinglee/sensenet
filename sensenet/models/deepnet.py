import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import IMAGE_PATH, NUMERIC, CATEGORICAL
from sensenet.accessors import get_output_exposition
from sensenet.layers.utils import propagate
from sensenet.layers.tree import ForestPreprocessor
from sensenet.layers.construct import layer_sequence, tree_preprocessor
from sensenet.preprocess.preprocessor import Preprocessor

def instantiate_inputs(model):
    preprocessors = model['preprocess']
    ptypes = [p['type'] for p in preprocessors]

    ncols = len(preprocessors)
    nstrings = ptypes.count(IMAGE_PATH) + ptypes.count(CATEGORICAL)

    Input = tf.keras.Input

    return {
        'numeric': Input((ncols,), dtype=tf.float32, name='numeric'),
        'string': Input((nstrings,), dtype=tf.string, name='string')
    }

def deepnet_model(model, extras):
    outex = get_output_exposition(model)
    preprocessor = Preprocessor(model, extras)
    trees = tree_preprocessor(model)

    if 'networks' in model:
        all_layer_sequences = [layer_sequence(net) for net in model['networks']]
        use_trees = [net['trees'] for net in model['networks']]
    elif model['layers']:
        all_layer_sequences = [layer_sequence(model)]
        use_trees = [trees is not None]
    else:
        all_layer_sequences = [[]]
        use_trees = [False]

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
