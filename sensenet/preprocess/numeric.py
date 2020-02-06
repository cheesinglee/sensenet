import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.layers.utils import constant

class NumericPreprocessor(tf.keras.layers.Layer):
    def __init__(self, preprocessor):
        super(NumericPreprocessor, self).__init__()
        self._moments = [preprocessor['mean'], preprocessor['stdev']]

        # This should only happen if the feature had a constant value
        # in training
        if self._moments[1] == 0:
            self._moments[1] = 1

    def build(self, input_shape):
        self._mean = constant(self._moments[0])
        self._stdev = constant(self._moments[1])

    def call(self, inputs):
        output = (inputs - self._mean) / self._stdev
        return tf.cast(tf.reshape(output, [-1, 1]), tf.float32)

class BinaryPreprocessor(tf.keras.layers.Layer):
    def __init__(self, preprocessor):
        super(BinaryPreprocessor, self).__init__()
        self._values = [preprocessor['zero_value'], preprocessor['one_value']]

    def build(self, input_shape):
        self._zero_value = constant(self._values[0])

    def call(self, inputs):
        output = tf.not_equal(inputs, self._zero_value)
        return tf.cast(tf.reshape(output, [-1, 1]), tf.float32)
