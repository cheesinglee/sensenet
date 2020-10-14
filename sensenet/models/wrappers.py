import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from sensenet.accessors import is_yolo_model
from sensenet.load import load_points
from sensenet.models.bounding_box import box_detector
from sensenet.models.deepnet import deepnet_model

def tflite_export(tf_model, model_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    tflite_model = converter.convert()

    if model_path is not None:
        with open(model_path, 'wb') as fout:
            fout.write(tflite_model)

    return tflite_model

class Deepnet(object):
    def __init__(self, model, settings):
        self._preprocessors = model['preprocess']
        self._model = deepnet_model(model, settings)

    def predict(self, points):
        pvec = load_points(self._preprocessors, points)
        return self._model.predict(pvec)

    def __call__(self, input_data):
        # TODO:  We should probably add assertions or something here
        # to make sure we're getting at least the right number of inputs
        if isinstance(input_data, list):
            if isinstance(input_data[0], (float, int, str)):
                # Single unwrapped instance
                return self.predict([input_data])
            else:
                # Properly wrapped instance
                return self.predict(input_data)
        elif isinstance(input_data, str):
            # Single image path or single text field
            prediction = self.predict([[input_data]])
        else:
            dtype = str(type(input_data))
            raise TypeError('Cannot predict on arguments of type "%s"' % dtype)

class ObjectDetector(object):
    def __init__(self, model, settings):
        self._unfiltered = settings.output_unfiltered_boxes
        self._model = box_detector(model, settings)
        self._classes = model['output_exposition']['values']

    def predict(self, points):
        return self._model.predict(points)

    def __call__(self, input_data):
        # Single wrapped instance
        if isinstance(input_data, list):
            prediction = self.predict([input_data])
        # Single image path
        elif isinstance(input_data, str):
            prediction = self.predict([[input_data]])
        # Pixel-valued ndarray input
        elif isinstance(input_data, np.ndarray) and len(input_data.shape) == 3:
            array = np.expand_dims(input_data, axis=0)
            prediction = self.predict(array)
        # Something else (tf.tensor or python list)
        else:
            prediction = self.predict(input_data)

        if self._unfiltered:
            return prediction
        else:
            boxes, scores, classes = prediction
            output_boxes = []

            for box, score, cls in zip(boxes[0], scores[0], classes[0]):
                output_boxes.append({
                    'box': [int(c) for c in box],
                    'label': self._classes[int(cls)],
                    'score': float(score)})

            return output_boxes

    def export(self, path):
        return tflite_export(self._model, path)

def is_deepnet(model):
    return 'preprocess' in model and ('layers' in model or 'networks' in model)

def create_model(model, settings=None):
    if is_deepnet(model):
        if is_yolo_model(model):
            return ObjectDetector(model, settings)
        else:
            return Deepnet(model, settings)
    else:
        raise ValueError('Model format not recognized: %s' % str(model.keys()))
