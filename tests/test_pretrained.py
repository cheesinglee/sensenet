import json
import gzip

import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import CATEGORICAL, IMAGE_PATH, BOUNDING_BOX
from sensenet.pretrained import get_pretrained_network, get_pretrained_readout
from sensenet.pretrained import cnn_resource_path
from sensenet.load import load_points
# from sensenet.graph.construct import make_layers
from sensenet.graph.classifier import deepnet_model
# from sensenet.graph.image import image_preprocessor, complete_image_network
# from sensenet.graph.bounding_box import box_detector, image_projector

EXTRA_PARAMS = {'path_prefix': 'tests/data/images/'}

def create_image_model(name, bb_thld=None):
    pretrained = get_pretrained_network(name)
    extras = dict(EXTRA_PARAMS)

    if 'yolo' in name:
        pretrained = complete_image_network(pretrained)
        pretrained['layers'] += get_pretrained_readout(pretrained)
        otype = BOUNDING_BOX
        nclasses = 80
        layers = None
        extras['bounding_box_threshold'] = bb_thld
    else:
        otype = CATEGORICAL
        nclasses = 1000
        layers = get_pretrained_readout(pretrained)

    network = {
        'output_exposition': {'type': otype, 'values': [None] * nclasses},
        'layers': layers,
        'trees': None,
        'image_network': pretrained,
        'preprocess': [{'type': IMAGE_PATH, 'index': 0}]
    }

    print([l['type'] for l in layers])
    print([l.keys() for l in layers])
    print([l['activation_function'] for l in layers])
    return network, deepnet_model(network, extras)

def classify(network_name, accuracy_threshold):
    network, image_model = create_image_model(network_name)

    for image, cidx in [('dog.jpg', 254), ('bus.jpg', 779)]:
        point = load_points(network, [[image]])
        pred = image_model.predict(point)

        print(len(pred.flatten().tolist()))

        for i, p in enumerate(pred.flatten().tolist()):
            if i == cidx:
                assert p > accuracy_threshold, str((i, p))
            else:
                assert p < 0.01, str((i, p))


def test_resnet50():
    classify('resnet50', 0.99)

def test_mobilenet():
    classify('mobilenet', 0.97)

def test_xception():
    classify('xception', 0.88)

def test_resnet18():
    classify('resnet18', 0.97)

def test_mobilenetv2():
    classify('mobilenetv2', 0.88)

def detect_bounding_boxes(network_name, nboxes, class_list, threshold):
    network, variables = create_image_model(network_name, threshold)

    with tf.Session() as sess:
        detector = image_projector(variables, 'bounding_box_preds', sess)
        boxes, scores, classes = detector('pizza_people.jpg')

    assert len(boxes) == nboxes
    assert sorted(set(classes)) == sorted(class_list), str(set(classes))

def test_yolov3():
    detect_bounding_boxes('yolov3', 6, [0, 60, 53], 0.6)

def test_tinyyolov3():
    detect_bounding_boxes('tinyyolov3', 3, [0, 53], 0.3)
