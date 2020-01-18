import json
import gzip

import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import CATEGORICAL, IMAGE_PATH, BOUNDING_BOX
from sensenet.pretrained import get_pretrained_network, get_pretrained_readout
from sensenet.pretrained import cnn_resource_path
from sensenet.graph.construct import make_layers
from sensenet.graph.classifier import initialize_variables, create_network
from sensenet.graph.image import image_preprocessor, complete_image_network
from sensenet.graph.bounding_box import box_detector, image_projector

TEST_DIR = 'tests/data/images/'

def project_and_classify(network_name, accuracy_threshold):
    pretrained = get_pretrained_network(network_name)

    network = {
        'output_exposition': {'type': CATEGORICAL, 'values': [None] * 1000},
        'layers': get_pretrained_readout(pretrained),
        'trees': None,
        'image_network': pretrained,
        'preprocess': [{'type': IMAGE_PATH}]
    }

    variables = initialize_variables(network)
    variables.update(image_preprocessor(variables, network, True, TEST_DIR))

    noutputs = network['image_network']['metadata']['outputs']

    with tf.Session() as sess:
        projector = image_projector(variables, 'image_preds', sess)

        X = tf.placeholder(tf.float32, shape=(None, noutputs))
        variables['preprocessed_X'] = X

        outputs = create_network(network, variables)

        sess.run(tf.global_variables_initializer())
        Xdog = projector('dog.jpg').tolist()
        Xbus = projector('bus.jpg').tolist()

        assert len(Xdog[0]) == len(Xbus[0]), str((len(Xdog[0]), len(Xbus[0])))
        assert noutputs == len(Xdog[0]), str((noutputs, len(Xdog[0])))

        for Xin, cidx in [(Xdog, 254), (Xbus, 779)]:
            pred = outputs.eval({X: Xin})

            for i, p in enumerate(pred.flatten().tolist()):
                if i == cidx:
                    assert p > accuracy_threshold, str((i, p))
                else:
                    assert p < 0.01


def test_resnet50():
    project_and_classify('resnet50', 0.99)

def test_mobilenet():
    project_and_classify('mobilenet', 0.97)

def test_xception():
    project_and_classify('xception', 0.88)

def test_resnet18():
    project_and_classify('resnet18', 0.97)

def test_mobilenetv2():
    project_and_classify('mobilenetv2', 0.88)

def detect_bounding_boxes(network_name, nboxes, class_list, thresh):
    pretrained = complete_image_network(get_pretrained_network(network_name))
    pretrained['layers'] += get_pretrained_readout(pretrained)

    network = {
        'output_exposition': {'type': BOUNDING_BOX, 'values': [None] * 80},
        'layers': None,
        'trees': None,
        'image_network': pretrained,
        'preprocess': [{'type': IMAGE_PATH}]
    }

    variables = initialize_variables(network)
    variables.update(box_detector(variables, network, True, TEST_DIR, thresh))

    with tf.Session() as sess:
        detector = image_projector(variables, 'bounding_box_preds', sess)
        boxes, scores, classes = detector('pizza_people.jpg')

    assert len(boxes) == nboxes
    assert sorted(set(classes)) == sorted(class_list), str(set(classes))

def test_yolov3():
    detect_bounding_boxes('yolov3', 6, [0, 60, 53], 0.6)

def test_tinyyolov3():
    detect_bounding_boxes('tinyyolov3', 3, [0, 53], 0.3)
