import collections

import sensenet.importers
tf = sensenet.importers.import_tensorflow()
np = sensenet.importers.import_numpy()

from sensenet.layers.utils import constant, transpose

def to_graph(root, inputs, noutputs):
    node_stack = collections.deque()
    mask_stack = collections.deque()

    tree = root
    nrows = tf.shape(inputs)[0]
    current_mask = tf.tile([True], (nrows,))
    all_idxs = tf.range(nrows, dtype=tf.int32)
    outputs = tf.zeros((nrows, noutputs), dtype=tf.float32)

    finished = False

    while not finished:

        while len(tree) > 2:
            split_index = constant(tree[0], tf.int32)
            split_value = constant(tree[1])

            less = inputs[:,split_index] <= split_value
            left = tf.logical_and(current_mask, less)
            right = tf.logical_and(current_mask, tf.math.logical_not(less))

            node_stack.append(tree)
            mask_stack.append(right)

            tree = tree[2]
            current_mask = left

        out_idxs = tf.where(current_mask)
        node_output = tf.reshape(constant(tree[0]), [1, -1])
        out_preds = tf.tile(node_output, [tf.size(out_idxs), 1])
        outputs = tf.tensor_scatter_nd_update(outputs, out_idxs, out_preds)

        if node_stack:
            tree = node_stack.pop()
            tree = tree[3]
            current_mask = mask_stack.pop()
        else:
            finished = True

    return outputs

class DecisionNode(tf.keras.layers.Layer):
    def __init__(self, tree, noutputs):
        super(DecisionNode, self).__init__()

        assert len(tree) == 2 and tree[-1] is None
        self._outputs = tree[0]

    def build(self, input_shape):
        self._output_tensor = tf.reshape(constant(self._outputs), [1, -1])

    def call(self, inputs):
        return tf.tile(self._output_tensor, [tf.shape(inputs)[0], 1])

class DecisionTree(tf.keras.layers.Layer):
    def __init__(self, tree, noutputs):
        super(DecisionTree, self).__init__()

        self._tree = tree
        self._noutputs = noutputs

    def call(self, inputs):
        return to_graph(self._tree, inputs, self._noutputs)

class DecisionForest(tf.keras.layers.Layer):
    def __init__(self, trees):
        super(DecisionForest, self).__init__()

        # Get number of outputs
        leaf_node = trees[0]

        while len(leaf_node) > 2:
            leaf_node = leaf_node[2]

        assert len(leaf_node) == 2 and leaf_node[-1] is None
        noutputs = len(leaf_node[0])

        self._trees = []

        for tree in trees:
            if len(tree) == 2 and tree[-1] is None:
                self._trees.append(DecisionNode(tree, noutputs))
            else:
                self._trees.append(DecisionTree(tree, noutputs))

    def call(self, inputs):
        all_preds = []
        nrows = tf.shape(inputs)[0]

        for tree in self._trees:
            preds = tree(inputs)
            all_preds.append(preds)

        summed = tf.math.accumulate_n(all_preds)

        return summed / len(all_preds)

class ForestPreprocessor(tf.keras.layers.Layer):
    def __init__(self, model):
        super(ForestPreprocessor, self).__init__()

        self._forests = []
        self._ranges = []

        for input_range, trees in model['trees']:
            self._forests.append([input_range, DecisionForest(trees)])

    def call(self, inputs):
        all_preds = []

        for input_range, forest in self._forests:
            start, end = input_range
            tree_inputs = inputs[:,start:end]
            all_preds.append(forest(tree_inputs))

        return tf.concat(all_preds + [inputs], -1)
