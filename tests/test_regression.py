import json
import gzip

import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import NUMERIC, CATEGORICAL
from sensenet.load import load_points
from sensenet.graph.classifier import initialize_variables
from sensenet.graph.classifier import create_classifier, create_preprocessor

SIMPLE_REGRESSIONS = 'tests/data/regression.json.gz'
EMBEDDING_REGRESSIONS = 'tests/data/embedding.json.gz'

def read_regression(path):
    with gzip.open(path, "rb") as fin:
        return json.loads(fin.read().decode('utf-8'))

def single_simple(index):
    atest = read_regression(SIMPLE_REGRESSIONS)[index]
    model, test_points = atest['model'], atest['validation']
    variables = create_classifier(model)

    Xin = variables['raw_X']
    outputs = variables['network_outputs']

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

def fake_outex(test_info):
    anode = test_info['trees'][0][1][0]
    while len(anode) > 2:
        anode = anode[2]

    nouts = len(anode[0])

    if nouts > 1:
        return {'type': CATEGORICAL, 'values': ['v%d' for v in range(nouts)]}
    else:
        return {'type': NUMERIC, 'mean': 0, 'stdev': 1}

def test_embedding():
    artifact = read_regression(EMBEDDING_REGRESSIONS)

    for test_info in artifact:
        test_info['output_exposition'] = fake_outex(test_info)
        inputs = load_points(test_info, test_info['input_data'])['input_X']

        variables = initialize_variables(test_info)
        Xin = variables['raw_X']

        pvars = create_preprocessor(test_info, variables)
        processed = pvars['preprocessed_X']
        embedded = pvars['embedded_X']

        with tf.Session() as sess:
            result = sess.run([processed, embedded], feed_dict={Xin: inputs})

            data_with_trees = np.array(test_info['with_trees'])
            data_without_trees = np.array(test_info['without_trees'])

            assert np.allclose(result[0], data_without_trees)
            assert np.allclose(result[1], data_with_trees)

def test_simple_search():
    pass
