import sensenet.importers
tf = sensenet.importers.import_tensorflow()

class CategoricalPreprocessor(tf.keras.layers.Layer):
    def __init__(self, preprocessor):
        super(CategoricalPreprocessor, self).__init__()
        self._depth = len(preprocessor['values'])

    def call(self, inputs):
        output = tf.one_hot(tf.cast(inputs, tf.int32), self._depth)
        return tf.cast(output, tf.float32)
