import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

import json
import gzip

from sensenet.constants import CATEGORICAL, IMAGE_PATH, BOUNDING_BOX

from sensenet.accessors import get_image_shape
from sensenet.load import load_points
from sensenet.models.image import pretrained_image_model, image_feature_extractor
from sensenet.models.image import image_layers, get_pretrained_network
from sensenet.models.settings import Settings, ensure_settings
from sensenet.preprocess.image import make_image_reader

from .utils import TEST_DATA_DIR, TEST_IMAGE_DATA

EXTRA_PARAMS = {
    'bounding_box_threshold': 0.5,
    'image_path_prefix': TEST_IMAGE_DATA,
    'input_image_format': 'file',
    'load_pretrained_weights': True
}

def create_image_model(network_name, additional_settings):
    extras = dict(EXTRA_PARAMS)

    if additional_settings:
        extras.update(additional_settings)

    return pretrained_image_model(network_name, Settings(extras))

def reader_for_network(network_name, additional_settings):
    extras = dict(EXTRA_PARAMS)

    if additional_settings:
        extras.update(additional_settings)

    image_shape = get_image_shape(get_pretrained_network(network_name))
    return make_image_reader(Settings(extras), image_shape, False)

def check_image_prediction(prediction, index, pos_threshold, neg_threshold):
    for i, p in enumerate(prediction.flatten().tolist()):
        if i == index:
            assert p > pos_threshold, str((i, p))
        else:
            assert p < neg_threshold, str((i, p))

def classify(network_name, accuracy_threshold):
    pixel_input = {'input_image_format': 'pixel_values'}

    network = get_pretrained_network(network_name)
    nlayers = len(network['image_network']['layers'])
    noutputs = network['image_network']['metadata']['outputs']
    preprocessors = network['preprocess']

    image_model = create_image_model(network_name, None)
    pixel_model = create_image_model(network_name, pixel_input)
    read = reader_for_network(network_name, None)

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
            check_image_prediction(pred, cidx, accuracy_threshold, 0.02)

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
    file_input = {'bounding_box_threshold': threshold}

    pixel_input = {
        'input_image_format': 'pixel_values',
        'bounding_box_threshold': threshold
    }

    image_detector = create_image_model(network_name, file_input)
    pixel_detector = create_image_model(network_name, pixel_input)
    read = reader_for_network(network_name, {'rescale_type': 'pad'})

    file_pred = image_detector.predict([['pizza_people.jpg']])
    img_px = np.expand_dims(read('pizza_people.jpg').numpy(), axis=0)
    pixel_pred = pixel_detector.predict(img_px)

    for pred in [file_pred, pixel_pred]:
        boxes, scores, classes = pred[0][0], pred[1][0], pred[2][0]

        assert len(boxes) == len(scores) == nboxes, len(boxes)
        assert sorted(set(classes)) == sorted(class_list), classes

def test_tinyyolov4():
    detect_bounding_boxes('tinyyolov4', 5, [0, 53], 0.4)

def test_empty():
    detector = create_image_model('tinyyolov4', None)
    boxes, scores, classes  = detector.predict([['black.png']])

    assert len(boxes[0]) == 0
    assert len(scores[0]) == 0
    assert len(classes[0]) == 0

def test_scaling():
    detector = create_image_model('tinyyolov4', None)
    boxes, scores, classes  = detector.predict([['strange_car.png']])

    assert 550 < boxes[0, 0, 0] < 600,  boxes[0, 0]
    assert 220 < boxes[0, 0, 1] < 270,  boxes[0, 0]
    assert 970 < boxes[0, 0, 2] < 1020,  boxes[0, 0]
    assert 390 < boxes[0, 0, 3] < 440,  boxes[0, 0]

    assert scores[0] > 0.9
    assert classes[0] == 2
