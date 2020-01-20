import json
import gzip

import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import NUMERIC, CATEGORICAL
from sensenet.load import load_points
from sensenet.graph.classifier import initialize_variables
from sensenet.graph.classifier import create_classifier, create_preprocessor

EMBEDDING_REGRESSION = 'tests/data/embedding.json.gz'
SIMPLE_REGRESSIONS = 'tests/data/regression.json.gz'
SEARCH_REGRESSION = 'tests/data/search_regression.json.gz'
IMAGE_REGRESSION = 'tests/data/image_regression.json.gz'

EXTRA_PARAMS = {'path_prefix': 'tests/data/images/digits/'}

def read_regression(path):
    with gzip.open(path, "rb") as fin:
        return json.loads(fin.read().decode('utf-8'))

def make_feed(variables, inputs):
    return {variables[k]: inputs[k] for k in inputs.keys()}

def validate_predictions(test_artifact):
    model, test_points = [test_artifact[k] for k in ['model', 'validation']]
    variables = create_classifier(model, EXTRA_PARAMS)

    lists = [t['input'] for t in test_points]
    inputs = load_points(model, lists)
    outputs = variables['network_outputs']

    ins = load_points(model, [t['input'] for t in test_points])
    outs = np.array([t['output'] for t in test_points])

    with tf.Session() as sess:
        for i, true_pred in enumerate(outs):
            point = {k: np.reshape(ins[k][i,:], [1, -1]) for k in ins.keys()}
            mod_pred = outputs.eval(make_feed(variables, point))
            assert np.allclose(mod_pred[0], true_pred, atol=1e-7)

        mod_preds = outputs.eval(make_feed(variables, inputs))
        assert np.allclose(mod_preds, outs, atol=1e-7)

def single_simple(index):
    test_artifact = read_regression(SIMPLE_REGRESSIONS)[index]
    validate_predictions(test_artifact)

def test_simple_networks():
    rdata = read_regression(SIMPLE_REGRESSIONS)

    for i in range(len(rdata)):
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

def single_embedding(index):
    test_info = read_regression(EMBEDDING_REGRESSION)[index]
    test_info['output_exposition'] = fake_outex(test_info)
    inputs = load_points(test_info, test_info['input_data'])

    variables = initialize_variables(test_info, None)
    pvars = create_preprocessor(test_info, variables)
    outputs = [pvars['preprocessed_X'], pvars['embedded_X']]

    with tf.Session() as sess:
        result = sess.run(outputs, feed_dict=make_feed(variables, inputs))

        data_with_trees = np.array(test_info['with_trees'])
        data_without_trees = np.array(test_info['without_trees'])

        assert np.allclose(result[0], data_without_trees)
        assert np.allclose(result[1], data_with_trees)

def test_embedding():
    artifact = read_regression(EMBEDDING_REGRESSION)

    for i in range(len(artifact)):
        yield single_embedding, i

def test_search():
    test_artifact = read_regression(SEARCH_REGRESSION)[0]
    validate_predictions(test_artifact)

def test_images():
    test_artifact = read_regression(IMAGE_REGRESSION)[0]
    validate_predictions(test_artifact)
