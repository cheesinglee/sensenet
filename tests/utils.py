import os
import gzip
import json

import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.layers.utils import propagate

TEST_DATA_DIR = 'tests/data/'

def read_regression(path, root=TEST_DATA_DIR):
    with gzip.open(os.path.join(root, path), "rb") as fin:
        return json.loads(fin.read().decode('utf-8'))

def make_model(lseq, ninputs):
    inputs = tf.keras.Input((ninputs,), dtype=tf.float32)
    outputs = propagate(lseq, inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
