import json
import gzip

import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from sensenet.load import load_points
from sensenet.graph.classifier import create_topology

SIMPLE_REGRESSIONS = 'tests/data/regression.json.gz'

def read_regression(path):
    with gzip.open(path, "rb") as fin:
        return json.loads(fin.read().decode('utf-8'))

def single_simple(index):
    atest = read_regression(SIMPLE_REGRESSIONS)[index]
    model, test_points = atest['model'], atest['validation']
    Xin, outputs = create_topology(model)

    ins = load_points(model, [t['input'] for t in test_points])['input_X']
    outs = np.array([t['output'] for t in test_points])

    with tf.Session() as sess:
        for in_point, true_pred in zip(ins, outs):
            mod_pred = outputs.eval({Xin: np.reshape(in_point, [1, -1])})
            assert np.allclose(mod_pred[0], true_pred)

def test_simple_networks():
    rdata = read_regression(SIMPLE_REGRESSIONS)

    for i in [2]: # range(len(rdata)):
        yield single_simple, i

def test_simple_search():
    pass
