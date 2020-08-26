import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.accessors import get_image_shape
from sensenet.layers.construct import LAYER_FUNCTIONS
from sensenet.layers.utils import make_sequence, propagate

class YoloTrunk(tf.keras.layers.Layer):
    def __init__(self, network, nclasses):
        super(YoloTrunk, self).__init__()

        self._trunk = []
        self._concatenations = {}

        for i, layer in enumerate(network['layers'][:-1]):
            ltype = layer['type']
            self._trunk.append(LAYER_FUNCTIONS[ltype](layer))

            if ltype == 'concatenate':
                self._concatenations[i] = layer['inputs']

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

        return outputs

class YoloBranches(tf.keras.layers.Layer):
    def __init__(self, network, nclasses):
        super(YoloBranches, self).__init__()

        self._nclasses = nclasses
        self._input_size = get_image_shape(network)[1]
        self._branches = []

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

        return conv_output, tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def call(self, inputs):
        predictions = []

        for i, decoding_info, layers in self._branches:
            features = propagate(layers, inputs[i])
            predicted_values = self.decode_outputs(features, decoding_info)

            predictions.append(predicted_values)

        return predictions
