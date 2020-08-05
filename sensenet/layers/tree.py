import collections

import sensenet.importers
tf = sensenet.importers.import_tensorflow()
np = sensenet.importers.import_numpy()
tl = sensenet.importers.import_bigml_treelib()

from sensenet.layers.utils import constant

def into_arrays(node, all_nodes, outputs):
    if len(node) == 2 and node[-1] is None:
        all_nodes.append([-1, -1, len(outputs), len(outputs)])
        outputs.append(list(node[0]))
    else:
        this_node = [node[0], node[1], 0, 0]
        all_nodes.append(this_node)

        this_node[2] = len(all_nodes)
        into_arrays(node[2], all_nodes, outputs)

        this_node[3] = len(all_nodes)
        into_arrays(node[3], all_nodes, outputs)

class DecisionNode(tf.keras.layers.Layer):
    def __init__(self, tree):
        super(DecisionNode, self).__init__()

        assert len(tree) == 2 and tree[-1] is None
        self._outputs = tree[0]

    def build(self, input_shape):
        self._output_tensor = tf.reshape(constant(self._outputs), [1, -1])

    def call(self, inputs):
        return tf.tile(self._output_tensor, [tf.shape(inputs)[0], 1])

class DecisionTree(tf.keras.layers.Layer):
    def __init__(self, tree):
        super(DecisionTree, self).__init__()

        node_list = []
        outputs = []

        into_arrays(tree, node_list, outputs)

        self._split_indices = constant([n[0] for n in node_list], tf.int32)
        self._split_values = constant([n[1] for n in node_list])
        self._left = constant([n[2] for n in node_list], tf.int32)
        self._right = constant([n[3] for n in node_list], tf.int32)
        self._outputs = constant(outputs)

    def call(self, inputs):
        out_idxs = tl.BigMLTreeify(points=inputs,
                                   split_indices=self._split_indices,
                                   split_values=self._split_values,
                                   left=self._left,
                                   right=self._right)

        return tf.gather(self._outputs, tf.reshape(out_idxs, (-1,)))

class DecisionForest(tf.keras.layers.Layer):
    def __init__(self, trees):
        super(DecisionForest, self).__init__()

        self._trees = []

        for tree in trees:
            if len(tree) == 2 and tree[-1] is None:
                self._trees.append(DecisionNode(tree))
            else:
                self._trees.append(DecisionTree(tree))

    def call(self, inputs):
        all_preds = []

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
