import sensenet.importers

tf = sensenet.importers.import_tensorflow()
np = sensenet.importers.import_numpy()

from sensenet.graph.layers.tree import to_node_list, nodes_to_tensor

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
        [5.0, 90000, 222.22, -5.0, -50.0, -42.0],
        [5.001, -10000.0, -10000.0, 0.01, 0.01, -0.042],
        [6.0, 10000.0, 0.5, 45.32, 0.06, -0.42],
        [4567.0, -100.0, 0.77, -50.0, 0.0, 4.2],
        [123456.789, 100.0, 100.0, 0.0004, 99.99, -420.0]
    ]

    results = np.array([
        test_tree[2][2][2][0],
        test_tree[2][2][3][0],
        test_tree[2][3][2][0],
        test_tree[2][3][3][0],
        test_tree[3][2][2][0],
        test_tree[3][2][3][0],
        test_tree[3][3][2][0],
        test_tree[3][3][3][0]
    ], dtype=np.float32)

    nodes = to_node_list(test_tree, len(results[0]), 0)
    nodes.sort(key=lambda x: x['node_id'])

    assert [n['node_id'] for n in nodes] == list(range(len(nodes)))

    with tf.Session() as sess:
        Xin = tf.placeholder(tf.float32, shape=(None, 6), name='input_data')
        pred_graph = nodes_to_tensor(Xin, nodes)

        sess.run(tf.global_variables_initializer())
        preds = pred_graph.eval({Xin: np.array(test_points)})

    assert np.array_equal(preds, results)
