import json
import gzip

import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from sensenet.pretrained import get_pretrained_network, get_pretrained_readout
from sensenet.pretrained import cnn_resource_path
from sensenet.graph.construct import make_layers
from sensenet.graph.classifier import create_placeholders, create_dense_layers
from sensenet.graph.image import image_preprocessor, make_loader
from sensenet.graph.bounding_box import box_detector, box_projector

def image_file_projector(loader, variables, tf_session):
    X = variables['image_X']
    preds = variables['image_preds']
    is_training = variables['is_training']
    keep_prob = variables['keep_prob']

    batch_params = {keep_prob: 1.0, is_training: False}

    def proj(image_path_s):
        if isinstance(image_path_s, list):
            if len(image_path_s) > 1:
                imgs = [loader(path) for path in image_path_s]
                batch_params[X] = np.squeeze(np.asarray(imgs))
            else:
                batch_params[X] = loader(image_path_s[0])
        else:
            batch_params[X] = loader(image_path_s)

        return tf_session.run(preds, feed_dict=batch_params)

    return proj

def project_and_classify(network_name, accuracy_threshold):
    variables = create_placeholders()

    network = get_pretrained_network(network_name)
    readout = get_pretrained_readout(network)
    variables.update(image_preprocessor(network, 1, variables))

    loader = make_loader(network)
    noutputs = network['metadata']['outputs']

    with tf.Session() as sess:
        proj = image_file_projector(loader, variables, sess)

        X = tf.placeholder(tf.float32, shape=(None, noutputs))
        outputs = create_dense_layers(X, readout, variables)

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


def test_resnet():
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
    variables = create_placeholders()

    network = get_pretrained_network(network_name)
    readout = get_pretrained_readout(network)
    loader = make_loader(network)
    variables.update(box_detector(network, readout, variables, 80, threshold))

    with tf.Session() as sess:
        detector = box_projector(loader, variables, sess)

        sess.run(tf.global_variables_initializer())
        boxes, scores, classes = detector(test_path)

    assert len(boxes) == nboxes
    assert sorted(set(classes)) == sorted(class_list), str(set(classes))

def test_yolov3():
    detect_bounding_boxes('yolov3', 6, [0, 60, 53], 0.6)

def test_tinyyolov3():
    detect_bounding_boxes('tinyyolov3', 3, [0, 53], 0.3)
