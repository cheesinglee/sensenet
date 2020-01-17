import json
import gzip

import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import CATEGORICAL, IMAGE_PATH
from sensenet.pretrained import get_pretrained_network, get_pretrained_readout
from sensenet.pretrained import cnn_resource_path
from sensenet.graph.construct import make_layers
from sensenet.graph.classifier import initialize_variables, create_network
from sensenet.graph.image import image_preprocessor
from sensenet.graph.bounding_box import box_detector, box_projector

def image_file_projector(variables, tf_session):
    X = variables['image_paths']
    preds = variables['image_preds']

    def proj(image_path):
        return tf_session.run(preds, feed_dict={X: np.array([[image_path]])})

    return proj

def project_and_classify(network_name, accuracy_threshold):
    prefix = 'tests/data/images/'
    network = get_pretrained_network(network_name)
    readout = get_pretrained_readout(network)

    variables = initialize_variables({'preprocess': [{'type': IMAGE_PATH}]})
    variables.update(image_preprocessor(variables, network, True, prefix))

    noutputs = network['metadata']['outputs']

    with tf.Session() as sess:
        proj = image_file_projector(variables, sess)

        X = tf.placeholder(tf.float32, shape=(None, noutputs))
        variables['preprocessed_X'] = X

        readout_network = {
            'output_exposition': {'type': CATEGORICAL, 'values': [None] * 1000},
            'layers': readout,
            'trees': None
        }

        outputs = create_network(readout_network, variables)

        sess.run(tf.global_variables_initializer())
        Xdog = proj('dog.jpg').tolist()
        Xbus = proj('bus.jpg').tolist()

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

def detect_bounding_boxes(network_name, nboxes, class_list, threshold):
    test_path = 'tests/data/pizza_people.jpg'

    network = get_pretrained_network(network_name)
    readout = get_pretrained_readout(network)
    variables = box_detector(network, readout, 80, threshold)

    with tf.Session() as sess:
        detector = box_projector(read_fn(network, '.'), variables, sess)

        sess.run(tf.global_variables_initializer())
        boxes, scores, classes = detector(test_path)

    assert len(boxes) == nboxes
    assert sorted(set(classes)) == sorted(class_list), str(set(classes))

def test_yolov3():
    detect_bounding_boxes('yolov3', 6, [0, 60, 53], 0.6)

def test_tinyyolov3():
    detect_bounding_boxes('tinyyolov3', 3, [0, 53], 0.3)
