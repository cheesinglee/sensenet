import os
import gzip
import json

import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.layers.utils import propagate

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
