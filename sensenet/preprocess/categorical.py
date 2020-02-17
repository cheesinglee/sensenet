import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.layers.utils import constant

class CategoricalPreprocessor(tf.keras.layers.Layer):
    def __init__(self, preprocessor):
        super(CategoricalPreprocessor, self).__init__()
        self._values = constant([preprocessor['values']], tf.string)

    def call(self, inputs):
        output = tf.math.equal(tf.reshape(inputs, [-1, 1]), self._values)
        return tf.cast(output, tf.float32)
