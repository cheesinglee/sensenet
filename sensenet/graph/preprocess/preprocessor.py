import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import NUMERIC, CATEGORICAL, IMAGE_PATH

from sensenet.graph.preprocess.categorical import CategoricalPreprocessor
from sensenet.graph.preprocess.numeric import NumericPreprocessor
from sensenet.graph.preprocess.numeric import BinaryPreprocessor
from sensenet.graph.preprocess.image import ImagePreprocessor

class Preprocessor(tf.keras.layers.Layer):
    def __init__(self, model):
        super(Preprocessor, self).__init__()
        self._preprocessors = []

        if model.get('image_network', None) is not None:
            img_proc = ImageProcessor(model['image_network'])

        for pp in model['preprocess']:
            ptype = pp['type']

            if ptype == NUMERIC:
                if 'mean' in pp:
                    self._preprocessors.append(NumericPreprocessor(pp))
                elif 'zero_value' in pp:
                    self._preprocessors.append(BinaryPreprocessor(pp))
            elif ptype == CATEGORICAL:
                self._preprocessors.append(CategoricalPreprocessor(pp))
            elif ptype == IMAGE_PATH:
                self._preprocessors.append(img_proc)
            else:
                raise ValueError('Cannot make processor with type "%s"' % ptype)


    def call(self, inputs):
        processed = []

        for i, pp in enumerate(self._preprocessors):
            processed.append(pp(inputs[:,i]))

        if len(processed) > 1:
            return tf.concat(processed, -1)
        else:
            return processed[0]
