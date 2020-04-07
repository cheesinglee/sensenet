import json
import gzip

import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import CATEGORICAL, IMAGE_PATH, BOUNDING_BOX

from sensenet.load import load_points
from sensenet.pretrained import get_image_network, load_pretrained_weights
from sensenet.models.deepnet import deepnet_model
from sensenet.models.bounding_box import box_detector

EXTRA_PARAMS = {'path_prefix': 'tests/data/images/'}

def create_image_model(name, bb_thld=None):
    extras = dict(EXTRA_PARAMS)
    network = get_image_network(name)

    if 'yolo' in name:
        extras['bounding_box_threshold'] = bb_thld
        return network, box_detector(network, extras)
    else:
        return network, deepnet_model(network, extras)

def classify(network_name, accuracy_threshold):
    network, image_model = create_image_model(network_name)
    load_pretrained_weights(image_model, network['image_network'])

    for image, cidx in [('dog.jpg', 254), ('bus.jpg', 779)]:
        point = load_points(network, [[image]])
        pred = image_model.predict(point)

        for i, p in enumerate(pred.flatten().tolist()):
            if i == cidx:
                assert p > accuracy_threshold, str((i, p))
            else:
                assert p < 0.02, str((i, p))


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
    network, detector = create_image_model(network_name, threshold)
    load_pretrained_weights(detector, network['image_network'])

    boxes, scores, classes = detector.predict([['pizza_people.jpg']])

    assert len(boxes[0]) == nboxes
    assert sorted(set(classes[0])) == sorted(class_list), str(set(classes[0]))

def test_yolov3():
    detect_bounding_boxes('yolov3', 6, [0, 60, 53], 0.6)

def test_tinyyolov3():
    detect_bounding_boxes('tinyyolov3', 3, [0, 53], 0.4)
