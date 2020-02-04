import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from sensenet.graph.layers.utils import transpose
from sensenet.graph.layers.core_layers import dense, batchnorm, activation_function

def with_popped_activation(params):
    afn = activation_function(params)
    params_copy = dict(params)
    params_copy.pop('activation_function', None)

    return afn, params_copy

class LegacyBlock(tf.keras.layers.Layer):
    def __init__(self, params):
        super(LegacyBlock, self).__init__()

        dense_params, afn = with_popped_activation(params)

        self._dense = dense(dense_params)
        self._bnorm = batchnorm(params)
        self._activator = Activation(afn)

    def call(self, inputs):
        propigated = self._dense(inputs)
        normalized = self._bnorm(propigated)

        return self._activator(normalized)

def legacy(params):
    dense_params = dict(params)

    # Needs converting from old format
    dense_params['weights'] = transpose(dense_params['weights'])
    dense_params['beta'] = dense_params['offset']
    dense_params['gamma'] = dense_params['scale']

    if params['stdev'] is not None:
        variance = np.square(np.array(params['stdev'])) - 1e-3
        dense_params['variance'] = variance.tolist()
        dense_params['offset'] = np.zeros(np.array(variance.shape)).tolist()

        return LegacyBlock(params)
    else:
        return dense(dense_params)

class LegacyResidualBlock(tf.keras.layers.Layer):
    def __init__(self, params_list):
        super(LegacyResidualBlock, self).__init__()

        first_params = params_list[0]
        second_params, afn = with_popped_activation(params_list[1])

        self._first = legacy(first_params)
        self._second = legacy(second_params)
        self._activator = Activation(afn)

    def equalize_input_width(inputs, outputs):
        if inputs.shape[1] == outputs.shape[1]:
            return inputs
        elif inputs.shape[1] > outputs.shape[1]:
            return inputs[:,:outputs.shape[1]]
        else:  # need to repeat inputs until we get to the output shape
            to_concat = []
            ncols = 0

            while ncols < outputs.shape[1]:
                if ncols + inputs.shape[1] < outputs.shape[1]:
                    to_concat.append(inputs)
                    ncols += inputs.shape[1]
                else:
                    to_add = outputs.shape[1] - ncols
                    to_concat.append(inputs[:,:to_add])
                    ncols += to_add

            return tf.concat(to_concat, -1)

    def call(self, inputs):
        first_out = self._first(inputs)
        residuals = self._second(first_out)

        tiled_inputs = self.equalize_input_width(inputs, residuals)
        outputs = tiled_inputs + residuals

        return self._activator(outputs)

def make_legacy_sequence(layers_params):
    layers = []
    use_next = True

    for i, lp in enumerate(layers_params):
        if use_next:
            if i < len(layers_params) - 1:
                residuals = layers_params[i + 1].get('residuals', False)
            else:
                residuals = False

            if residuals:
                layer = LegacyResidualBlock([lp, layers_params[i + 1]])
                use_next = False
            else:
                layer = legacy(lp)

            layers.append(layer)
        else:
            use_next = True

    return layers
