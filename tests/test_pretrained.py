import json
import gzip

import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from sensenet.pretrained import get_pretrained_network, get_pretrained_readout
from sensenet.pretrained import cnn_resource_path
from sensenet.graph.construct import make_layers
from sensenet.graph.classifier import create_dense_layers
from sensenet.graph.image import image_preprocessor, read_fn
from sensenet.graph.bounding_box import box_detector, box_projector

def image_file_projector(loader, variables, tf_session):
    X = variables['image_X']
    preds = variables['image_preds']

    def proj(image_path):
        img_in = loader(image_path)
        # Add axes - one row and one image per row
        batch_params = {X: img_in[np.newaxis, np.newaxis, ...]}

        return tf_session.run(preds, feed_dict=batch_params)

    return proj

def project_and_classify(network_name, accuracy_threshold):
    network = get_pretrained_network(network_name)
    readout = get_pretrained_readout(network)
    variables = image_preprocessor(network, 1)

    noutputs = network['metadata']['outputs']

    with tf.Session() as sess:
        proj = image_file_projector(read_fn(network), variables, sess)

        X = tf.placeholder(tf.float32, shape=(None, noutputs))
        variables['preprocessed_X'] = X
        outputs = create_dense_layers(readout, variables, False)

        sess.run(tf.global_variables_initializer())
        Xdog = proj('tests/data/dog.jpg').tolist()
        Xbus = proj('tests/data/bus.jpg').tolist()

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
        detector = box_projector(read_fn(network), variables, sess)

        sess.run(tf.global_variables_initializer())
        boxes, scores, classes = detector(test_path)

    assert len(boxes) == nboxes
    assert sorted(set(classes)) == sorted(class_list), str(set(classes))

def test_yolov3():
    detect_bounding_boxes('yolov3', 6, [0, 60, 53], 0.6)

def test_tinyyolov3():
    detect_bounding_boxes('tinyyolov3', 3, [0, 53], 0.3)
