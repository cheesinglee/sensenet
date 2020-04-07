import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.layers.utils import make_sequence, propagate
from sensenet.layers.legacy import make_legacy_sequence
from sensenet.layers.tree import ForestPreprocessor
from sensenet.layers.block import SIMPLE_LAYERS, BLOCKS

LAYER_FUNCTIONS = {}
LAYER_FUNCTIONS.update(SIMPLE_LAYERS)
LAYER_FUNCTIONS.update(BLOCKS)

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
