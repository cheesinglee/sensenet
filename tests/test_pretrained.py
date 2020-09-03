import json
import gzip

import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import CATEGORICAL, IMAGE_PATH, BOUNDING_BOX

from sensenet.accessors import get_image_shape
from sensenet.load import load_points
from sensenet.models.image import pretrained_image_model, image_feature_extractor
from sensenet.models.image import image_layers, get_pretrained_network
from sensenet.models.settings import Settings
from sensenet.preprocess.image import get_image_reader_fn

EXTRA_PARAMS = {
    'image_path_prefix': 'tests/data/images/',
    'input_image_format': 'file',
    'load_pretrained_weights': True
}

def create_image_model(network_name, box_threshold, image_format):
    extras = dict(EXTRA_PARAMS)
    extras['input_image_format'] = image_format
    extras['bounding_box_threshold'] = box_threshold

    extras = Settings(extras) if box_threshold else extras
    return pretrained_image_model(network_name, extras)

def reader_for_network(network_name):
    image_shape = get_image_shape(get_pretrained_network(network_name))
    path_prefix = EXTRA_PARAMS['image_path_prefix']

    return get_image_reader_fn(image_shape, 'file', path_prefix)

def classify(network_name, accuracy_threshold):
    network = get_pretrained_network(network_name)
    nlayers = len(network['image_network']['layers'])
    noutputs = network['image_network']['metadata']['outputs']
    preprocessors = network['preprocess']

    image_model = create_image_model(network_name, None, 'file')
    pixel_model = create_image_model(network_name, None, 'pixel_values')
    read = reader_for_network(network_name)

    assert len(image_layers(pixel_model)) == nlayers

    # Just check if this is possible
    image_feature_extractor(pixel_model)
    ex_mod = image_feature_extractor(image_model)

    for image, cidx in [('dog.jpg', 254), ('bus.jpg', 779)]:
        point = load_points(preprocessors, [[image]])
        file_pred = image_model.predict(point)

        img_px = np.expand_dims(read(image).numpy(), axis=0)
        pixel_pred = pixel_model.predict(img_px)

        for pred in [file_pred, pixel_pred]:
            for i, p in enumerate(pred.flatten().tolist()):
                if i == cidx:
                    assert p > accuracy_threshold, str((i, p))
                else:
                    assert p < 0.02, str((i, p))

    outputs = ex_mod(load_points(preprocessors, [['dog.jpg'], ['bus.jpg']]))
    assert outputs.shape == (2, noutputs)

def test_resnet50():
    classify('resnet50', 0.99)

def test_mobilenet():
    classify('mobilenet', 0.97)

def test_xception():
    classify('xception', 0.88)

def test_resnet18():
    classify('resnet18', 0.96)

def test_mobilenetv2():
    classify('mobilenetv2', 0.88)

def detect_bounding_boxes(network_name, nboxes, class_list, threshold):
    image_detector = create_image_model(network_name, threshold, 'file')
    pixel_detector = create_image_model(network_name, threshold, 'pixel_values')
    read = reader_for_network(network_name)

    file_pred = image_detector.predict([['pizza_people.jpg']])
    img_px = np.expand_dims(read('pizza_people.jpg').numpy(), axis=0)
    pixel_pred = pixel_detector.predict(img_px)

    for pred in [file_pred, pixel_pred]:
        boxes, scores, classes = pred

        assert len(boxes) == len(scores) == nboxes, len(boxes)
        assert sorted(set(classes)) == sorted(class_list), classes

def test_tinyyolov4():
    detect_bounding_boxes('tinyyolov4', 5, [0, 53], 0.4)

def test_scaling():
    detector = create_image_model('tinyyolov4', 0.5, 'file')
    pred = detector.predict([['strange_car.png']])

    boxes, scores, classes = pred

    assert 550 < boxes[0][0] < 600, boxes[0]
    assert 220 < boxes[0][1] < 270, boxes[0]
    assert 970 < boxes[0][2] < 1020, boxes[0]
    assert 370 < boxes[0][3] < 420, boxes[0]

    assert scores[0] > 0.9
    assert classes[0] == 2
