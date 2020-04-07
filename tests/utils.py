import os
import gzip
import json

import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.layers.utils import propagate
from sensenet.layers.construct import layer_sequence
from sensenet.models.deepnet import deepnet_model
from sensenet.models.bounding_box import box_detector

TEST_DATA_DIR = 'tests/data/'

def read_regression(path, root=TEST_DATA_DIR):
    return read_zipped_json(os.path.join(root, path))

def read_zipped_json(path):
    with gzip.open(path, "rb") as fin:
        return json.loads(fin.read().decode('utf-8'))

def make_model(lseq, input_shape):
    inputs = tf.keras.Input(input_shape, dtype=tf.float32)
    outputs = propagate(lseq, inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def image_model(network):
    name = network['image_network']['metadata']['base_image_network']

    if 'yolo' in name:
        return box_detector(network, {})
    else:
        return deepnet_model(network, {})
