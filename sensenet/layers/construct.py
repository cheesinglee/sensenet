import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.layers.block import SIMPLE_LAYERS, BLOCKS
from sensenet.layers.convolutional import CONVOLUTIONAL_LAYERS, get_shape_params
from sensenet.layers.legacy import make_legacy_sequence
from sensenet.layers.tree import ForestPreprocessor
from sensenet.layers.utils import make_sequence, propagate, WEIGHT_INITIALIZERS

LAYER_FUNCTIONS = {}
LAYER_FUNCTIONS.update(SIMPLE_LAYERS)
LAYER_FUNCTIONS.update(BLOCKS)

WEIGHTED_LAYERS = set()
WEIGHTED_LAYERS.update((CONVOLUTIONAL_LAYERS.keys()))
WEIGHTED_LAYERS.update(['dense', 'batch_normalization'])

class YoloTail(tf.keras.layers.Layer):
    def __init__(self, network):
        super(YoloTail, self).__init__()

        self._trunk = []
        self._branches = []
        self._concatenations = {}

        for i, layer in enumerate(network['layers'][:-1]):
            ltype = layer['type']
            self._trunk.append(LAYER_FUNCTIONS[ltype](layer))

            if ltype == 'concatenate':
                self._concatenations[i] = layer['inputs']

        assert network['layers'][-1]['type'] == 'yolo_output_branches'
        out_branches = network['layers'][-1]

        for i, branch in enumerate(out_branches['output_branches']):
            idx = branch['input']
            layers = make_sequence(branch['convolution_path'], LAYER_FUNCTIONS)

            self._branches.append((idx, layers))

    def call(self, inputs):
        outputs = []
        next_inputs = inputs

        for i, layer in enumerate(self._trunk):
            if i in self._concatenations:
                inputs = self._concatenations[i]
                next_inputs = layer([outputs[j] for j in inputs])
            else:
                next_inputs = layer(next_inputs)

            outputs.append(next_inputs)

        return [propagate(layers, outputs[i]) for i, layers in self._branches]

def layer_sequence(model):
    layers_params = model['layers']

    if any(p.get('type', None) == None for p in layers_params):
        return make_legacy_sequence(layers_params)
    elif layers_params[-1]['type'] == 'yolo_output_branches':
        return YoloTail(model)
    else:
        return make_sequence(layers_params, LAYER_FUNCTIONS)

def tree_preprocessor(model):
    if model.get('trees', None):
        return ForestPreprocessor(model)
    else:
        return None

def get_n_nodes(params):
    if isinstance(params['weights'], str):
        return int(params['number_of_nodes'])
    elif 'stdev' in params:
        return len(params['weights'])
    else:
        return len(params['weights'][0])

def remove_weights(element):
    if isinstance(element, dict):
        edict  = dict(element)
        ltype = edict.get('type', None)

        if ltype in WEIGHTED_LAYERS or 'weights' in edict:
            if ltype == 'dense' or 'weights' in edict:
                edict['number_of_nodes'] = get_n_nodes(edict)
            elif ltype in CONVOLUTIONAL_LAYERS:
                nfilters, kdims = get_shape_params(edict)
                edict['number_of_filters'] = nfilters
                edict['kernel_dimensions'] = kdims

            for key in edict:
                if key in WEIGHT_INITIALIZERS and isinstance(edict[key], list):
                    edict[key] = WEIGHT_INITIALIZERS[key]

        for key in edict:
            if isinstance(edict[key], (dict, list, tuple)):
                edict[key] = remove_weights(edict[key])

        return edict
    elif isinstance(element, (list, tuple)):
        return [remove_weights(e) for e in element]
    else:
        return element
