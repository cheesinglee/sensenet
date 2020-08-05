import json
import sys
import time
import random

import sklearn.tree as sktree
import sklearn.ensemble as skens

import sensenet.importers

tf = sensenet.importers.import_tensorflow()
np = sensenet.importers.import_numpy()

from sensenet.layers.utils import constant
from sensenet.layers.tree import DecisionTree, DecisionForest

from .utils import read_zipped_json

def test_simple_tree_prediction():
    test_tree = [
        0,
        5,
        [
            1,
            50,
            [3, 500, [[0.1, 0.2, 0.7], None], [[0.7, 0.2, 0.1], None]],
            [0, 3, [[0.2, 0.3, 0.5], None], [[0.5, 0.3, 0.2], None]]
        ],
        [
            2,
            0.5,
            [4, 0.05, [[0.01, 0.02, 0.97], None], [[0.97, 0.01, 0.02], None]],
            [2, 1, [[0.5, 0.1, 0.4], None], [[0.4, 0.5, 0.1], None]]
        ]
    ]

    test_points = [
        [-1.5, -0.07, 0.0, 0.0, -10.0, 42.0],
        [3.0, 23.456, 50.0, 501.501, -100.0, 42.0],
        [3.0, 90, 22.22, 0.0, 0.0, 42.0],
        [5.001, -10000.0, -10000.0, 0.01, 0.01, -0.042],
        [5.0, 90000, 222.22, -5.0, -50.0, -42.0],
        [6.0, 10000.0, 0.5, 45.32, 0.06, -0.42],
        [4567.0, -100.0, 0.77, -50.0, 0.0, 4.2],
        [123456.789, 100.0, 100.0, 0.0004, 99.99, -420.0]
    ]

    results = np.array([
        test_tree[2][2][2][0],
        test_tree[2][2][3][0],
        test_tree[2][3][2][0],
        test_tree[3][2][2][0],
        test_tree[2][3][3][0],
        test_tree[3][2][3][0],
        test_tree[3][3][2][0],
        test_tree[3][3][3][0]
    ], dtype=np.float32)

    # nodes = to_node_list(test_tree, len(results[0]), 0)
    # nodes.sort(key=lambda x: x['node_id'])

    # assert [n['node_id'] for n in nodes] == list(range(len(nodes)))
    # leaves = to_leaf_list(test_tree, [], [])
    # paths = to_paths(leaves)
    # outputs = propagate(paths, np.array(test_points, dtype=np.float32))

    tree = DecisionTree(test_tree)
    preds1 = tree(constant(test_points))
    preds2 = tree(constant(test_points))

    assert np.array_equal(preds1, results), str((preds1, results))
    assert np.array_equal(preds2, results)

def load_dataset(filename):
    data = read_zipped_json(filename)
    rng = random.Random(0)
    rng.shuffle(data)

    X = [[float(v) for v in p[:-1]] for p in data]

    classes = [p[-1] for p in data]
    cvalues = sorted(set(classes))
    y = [cvalues.index(c) for c in classes]

    return np.array(X), np.array(y)

def tree_to_list(tree, node_id):
    if tree.children_left[node_id] == sktree._tree.TREE_LEAF:
        values = tree.value[node_id][0]

        if len(values) > 1:
            sum_values = float(sum(values))
            return [[v / sum_values for v in values], None]
        else:
            return [values.tolist(), None]
    else:
        feature = int(tree.feature[node_id])
        threshold = float(tree.threshold[node_id])
        left_child = tree_to_list(tree, tree.children_left[node_id])
        right_child = tree_to_list(tree, tree.children_right[node_id])

        return [feature, threshold, left_child, right_child]


def trees_to_list(ensemble):
    jensemble = []
    for model in ensemble.estimators_:
        jensemble.append(tree_to_list(model.tree_, 0))

    return jensemble

def test_predictions():
    X, y = load_dataset('tests/data/diabetes.json.gz')
    X = X.astype(np.float32)

    ensemble = skens.RandomForestClassifier(n_estimators=64, random_state=0)
    ensemble.fit(X, y)

    points = np.tile(X, (256, 1))
    assert points.shape[0] > 128000

    start = time.time()
    sk_preds = ensemble.predict_proba(points)
    sk_time = time.time() - start

    trees = trees_to_list(ensemble)
    forest = DecisionForest(trees)
    inten = tf.keras.Input((8,), dtype=tf.float32)
    outten = forest(inten)
    model = tf.keras.Model(inputs=inten, outputs=outten)

    start = time.time()
    mod_preds = model(points).numpy()
    tf_time = time.time() - start

    sys.stdout.write('sk: %.3f / tf: %.3f ... ' % (sk_time, tf_time))
    sys.stdout.flush()

    assert mod_preds.shape == sk_preds.shape

    # Not all of these are going to be right; there are strange floating
    # point things between scikit and Tensorflow, but it's not
    # important for this test.
    assert np.sum(mod_preds == sk_preds) > len(mod_preds.flatten()) * 0.99
