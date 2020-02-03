import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.accessors import number_of_classes
from sensenet.graph.layers.utils import constant

def to_node_list(tree, noutputs, start_idx):
    this_node = {'node_id': start_idx}

    if len(tree) == 2 and tree[-1] is None:
        this_node['outputs'] = tree[0]
        this_node['split_index'] = 0
        this_node['split_value'] = 0.0
        this_node['left'] = start_idx
        this_node['right'] = start_idx

        return [this_node]
    else:
        this_node['outputs'] = [0.0 for _ in range(noutputs)]
        this_node['split_index'] = tree[0]
        this_node['split_value'] = tree[1]

        left_idx = start_idx + 1
        left_nodes = to_node_list(tree[2], noutputs, left_idx)
        right_nodes = to_node_list(tree[3], noutputs, left_idx + len(left_nodes))

        this_node['left'] = left_nodes[0]['node_id']
        this_node['right'] = right_nodes[0]['node_id']

        return [this_node] + left_nodes + right_nodes

def create_tree_tensors(node_list):
    assert [n['node_id'] for n in node_list] == list(range(len(node_list)))

    tree_tens = {}

    for key in ['split_index', 'split_value', 'outputs']:
        tree_tens[key] = [node[key] for node in node_list]

    tree_tens['next_matrix'] = [[n['left'], n['right']] for n in node_list]
    tree_tens['is_leaf'] = [n['left'] == n['node_id'] for n in node_list]

    for k in tree_tens:
        assert len(tree_tens[k]) == len(node_list)

    return tree_tens

class DecisionNode(tf.keras.layers.Layer):
    def __init__(self, node_list):
        super(DecisionNode, self).__init__()

        assert len(node_list) == 1
        self._outputs = node_list[0]['outputs']

    def build(self, input_shape):
        self._output_tensor = tf.reshape(constant(self._outputs), [1, -1])

    def call(self, inputs):
        return tf.tile(self._output_tensor, [inputs.shape[0], 1])

class DecisionTree(tf.keras.layers.Layer):
    def __init__(self, node_list):
        super(DecisionTree, self).__init__()

        assert len(node_list) > 1
        self._tensors = create_tree_tensors(node_list)

    def build(self, input_shape):
        self._is_leaf = constant(self._tensors['is_leaf'], tf.bool)
        self._indexes = constant(self._tensors['split_index'], tf.int32)
        self._split_values = constant(self._tensors['split_value'], tf.float32)
        self._next_matrix = constant(self._tensors['next_matrix'], tf.int32)
        self._outputs = constant(self._tensors['outputs'], tf.float32)

        self._nouts = len(self._tensors['outputs'][0])

    def call(self, inputs):
        nrows = inputs.shape[0]

        xcoords = tf.range(nrows, dtype=tf.int32)
        zero_idxs = tf.zeros([nrows, 1], dtype=tf.int32)
        first_output = tf.ones([nrows, self._nouts], dtype=tf.float32)

        def loop_cond(nodes, _):
            gathered = tf.gather(self._is_leaf, nodes)
            reduced = tf.reduce_all(gathered)
            return tf.logical_not(reduced)

        def loop_body(nodes, _):
            sidxs = tf.reshape(tf.gather(self._indexes, nodes), [nrows])
            svals = tf.reshape(tf.gather(self._split_values, nodes), [nrows])

            value_coords = tf.stack([xcoords, sidxs], axis=1)
            values = tf.gather_nd(inputs, value_coords)
            side = tf.dtypes.cast(values > svals, tf.int32)
            node_coords = tf.stack([tf.reshape(nodes, [nrows]), side], axis=1)

            next_nodes = tf.gather_nd(self._next_matrix, node_coords)
            next_outputs = tf.gather(self._outputs, next_nodes)

            return (tf.reshape(next_nodes, [-1, 1]),
                    tf.reshape(next_outputs, [-1, self._nouts]))

        _, preds = tf.while_loop(loop_cond, loop_body, [zero_idxs, first_output])

        return preds

class DecisionForest(tf.keras.layers.Layer):
    def __init__(self, node_lists):
        super(DecisionForest, self).__init__()

        self._trees = []

        for node_list in node_lists:
            if len(node_list) > 1:
                self._trees.append(DecisionTree(node_list))
            else:
                self._trees.append(DecisionNode(node_list))

    def call(self, inputs):
        all_preds = [tree(inputs) for tree in self._trees]
        summed = tf.add_n(all_preds)

        return summed / len(all_preds)

class ForestPreprocessor(tf.keras.layers.Layer):
    def __init__(self, model):
        super(ForestPreprocessor, self).__init__()
        noutputs = number_of_classes(model)

        self._forests = []
        self._ranges = []

        for input_range, trees in model['trees']:
            node_lists = [to_node_list(t, noutputs, 0) for t in trees]
            self._forests.append([input_range, DecisionForest(node_lists)])

    def call(self, inputs):
        all_preds = []

        for input_range, forest in self._forests:
            start, end = input_range
            tree_inputs = inputs[:,start:end]
            all_preds.append(forest(tree_inputs))

        return tf.concat(all_preds + [inputs], -1)
