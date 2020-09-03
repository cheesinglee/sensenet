from sensenet.accessors import is_yolo_model
from sensenet.load import load_points
from sensenet.models.bounding_box import box_detector
from sensenet.models.deepnet import deepnet_model

class Deepnet(object):
    def __init__(self, model, settings):
        self._preprocessors = model['preprocess']
        self._model = deepnet_model(model, settings)

    def predict(self, points):
        pvec = load_points(self._preprocessors, points)
        return self._model.predict(pvec)

    def __call__(self, input_data):
        if isinstance(input_data, list):
            # Single unwrapped instance
            if isinstance(input_data[0], (float, int, str)):
                return self.predict([input_data])
            else:
                return self.predict(input_data)
        else:
            dtype = str(type(input_data))
            raise TypeError('Cannot predict on arguments of type "%s"' % dtype)

class ObjectDetector(object):
    def __init__(self, model, settings):
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
        else:
            prediction = self.predict(input_data)

        boxes, scores, classes = prediction
        output_boxes = []

        for box, score, cls in zip(boxes, scores, classes):
            output_boxes.append({
                'box': [int(c) for c in box],
                'label': self._classes[int(cls)],
                'score': float(score)})

        return output_boxes

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
