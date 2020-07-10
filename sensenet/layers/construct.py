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

        self._nclasses = nclasses
        self._input_size = get_image_shape(network)[1]
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
            d_info = [branch[k] for k in ['strides', 'anchors', 'xyscale']]
            layers = make_sequence(branch['convolution_path'], LAYER_FUNCTIONS)

            self._branches.append((idx, d_info, layers))

    def decode_outputs(self, features, decoding_info):
        strides, anchors, xyscale = decoding_info
        fsize = tf.shape(features)[0]
        osize = tf.constant(self._input_size // int(strides))
        ans = len(anchors)

        feature_shape = (fsize, osize, osize, ans, 5 + self._nclasses)
        conv_output = tf.reshape(features, feature_shape)

        all_outputs = tf.split(conv_output, (2, 2, 1, self._nclasses), axis=-1)
        raw_dxdy, raw_dwdh, raw_conf, raw_prob = all_outputs

        xy_grid = tf.meshgrid(tf.range(osize), tf.range(osize))
        xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)
        xy_grid = tf.expand_dims(xy_grid, axis=0)
        xy_grid = tf.cast(tf.tile(xy_grid, [fsize, 1, 1, ans, 1]), tf.float32)

        xy_correction = (tf.sigmoid(raw_dxdy) * xyscale) - 0.5 * (xyscale - 1)
        pred_xy = (xy_correction + xy_grid) * strides
        pred_wh = tf.exp(raw_dwdh) * tf.constant(anchors, dtype=tf.float32)
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(raw_conf)
        pred_prob = tf.sigmoid(raw_prob)

        return pred_xywh, pred_prob * pred_conf

    def call(self, inputs):
        outputs = []
        printed = False
        next_inputs = inputs

        for i, layer in enumerate(self._trunk):
            if i in self._concatenations:
                inputs = self._concatenations[i]
                next_inputs = layer([outputs[j] for j in inputs])
            else:
                next_inputs = layer(next_inputs)

            outputs.append(next_inputs)

        predictions = []

        for i, decoding_info, layers in self._branches:
            features = propagate(layers, outputs[i])
            predictions.append(self.decode_outputs(features, decoding_info))

        return predictions

def layer_sequence(model):
    layers_params = model['layers']

    if any(p.get('type', None) == None for p in layers_params):
        return make_legacy_sequence(layers_params)
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
