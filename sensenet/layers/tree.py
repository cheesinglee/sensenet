import collections

import sensenet.importers
tf = sensenet.importers.import_tensorflow()
np = sensenet.importers.import_numpy()

from sensenet.accessors import number_of_classes
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
    def __init__(self, trees, noutputs):
        super(DecisionForest, self).__init__()

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
        noutputs = number_of_classes(model)

        self._forests = []
        self._ranges = []

        for input_range, trees in model['trees']:
            self._forests.append([input_range, DecisionForest(trees, noutputs)])

    def call(self, inputs):
        all_preds = []

        for input_range, forest in self._forests:
            start, end = input_range
            tree_inputs = inputs[:,start:end]
            all_preds.append(forest(tree_inputs))

        return tf.concat(all_preds + [inputs], -1)

####################################################
# While loop:  iteration seems slow
####################################################
#
# def to_node_list(tree, noutputs, start_idx):
#     this_node = {'node_id': start_idx}

#     if len(tree) == 2 and tree[-1] is None:
#         this_node['outputs'] = tree[0]
#         this_node['split_index'] = 0
#         this_node['split_value'] = 0.0
#         this_node['left'] = start_idx
#         this_node['right'] = start_idx

#         return [this_node]
#     else:
#         this_node['outputs'] = [0.0 for _ in range(noutputs)]
#         this_node['split_index'] = tree[0]
#         this_node['split_value'] = tree[1]

#         left_idx = start_idx + 1
#         left_nodes = to_node_list(tree[2], noutputs, left_idx)
#         right_nodes = to_node_list(tree[3], noutputs, left_idx + len(left_nodes))

#         this_node['left'] = left_nodes[0]['node_id']
#         this_node['right'] = right_nodes[0]['node_id']

#         return [this_node] + left_nodes + right_nodes

# def create_tree_tensors(node_list):
#     assert [n['node_id'] for n in node_list] == list(range(len(node_list)))

#     tree_tens = {}

#     for key in ['split_index', 'split_value', 'outputs']:
#         tree_tens[key] = [node[key] for node in node_list]

#     tree_tens['next_matrix'] = [[n['left'], n['right']] for n in node_list]
#     tree_tens['is_leaf'] = [n['left'] == n['node_id'] for n in node_list]

#     for k in tree_tens:
#         assert len(tree_tens[k]) == len(node_list)

#     return tree_tens

# def tree_function(tensors):
#     is_leaf = constant(tensors['is_leaf'], tf.bool)
#     indexes = constant(tensors['split_index'], tf.int32)
#     split_values = constant(tensors['split_value'], tf.float32)
#     next_matrix = constant(tensors['next_matrix'], tf.int32)
#     outputs = constant(tensors['outputs'], tf.float32)

#     nouts = len(tensors['outputs'][0])

#     def loop_cond(inputs, xcoords, nodes, _):
#         gathered = tf.gather(is_leaf, nodes)
#         reduced = tf.reduce_all(gathered)
#         return tf.logical_not(reduced)

#     def loop_body(inputs, xcoords, nodes, _):
#         nrows = tf.shape(inputs)[0]
#         sidxs = tf.reshape(tf.gather(indexes, nodes), [nrows])
#         svals = tf.reshape(tf.gather(split_values, nodes), [nrows])

#         value_coords = tf.stack([xcoords, sidxs], axis=1)
#         values = tf.gather_nd(inputs, value_coords)
#         side = tf.dtypes.cast(values > svals, tf.int32)
#         node_coords = tf.stack([tf.reshape(nodes, [nrows]), side], axis=1)

#         next_nodes = tf.gather_nd(next_matrix, node_coords)
#         next_outputs = tf.gather(outputs, next_nodes)

#         return (inputs,
#                 xcoords,
#                 tf.reshape(next_nodes, [-1, 1]),
#                 tf.reshape(next_outputs, [-1, nouts]))

#     def get_output(inputs):
#         nrows = tf.shape(inputs)[0]

#         xcoords = tf.range(nrows, dtype=tf.int32)
#         zero_idxs = tf.zeros([nrows, 1], dtype=tf.int32)
#         first_output = tf.ones([nrows, nouts], dtype=tf.float32)

#         _, _, _, preds = tf.while_loop(loop_cond,
#                                        loop_body,
#                                        [inputs, xcoords, zero_idxs, first_output],
#                                        back_prop=False,
#                                        parallel_iterations=1)

#         return preds

#     return get_output


####################################################
# Ragged tensor paths; too much memory
####################################################
#
# def to_leaf_list(tree, leaves, path):
#     if len(tree) == 2 and tree[-1] is None:
#         leaves.append([path, tree[0]])
#     else:
#         sidx = tree[0]
#         sval = tree[1]

#         to_leaf_list(tree[2], leaves, path + [(False, (sidx, sval))])
#         to_leaf_list(tree[3], leaves, path + [(True, (sidx, sval))])

#     return leaves

# def to_paths(tree):
#     leaf_list = to_leaf_list(tree, [], [])
#     split_list = []

#     for splits, output in leaf_list:
#         for side, split in splits:
#             split_list.append(split)

#     split_set = sorted(set(split_list))
#     split_dict = {s: i for i, s in enumerate(split_set)}
#     split_indices = [split[0] for split in split_set]
#     split_values = [split[1] for split in split_set]
#     splits_length = len(split_set)

#     outputs = []
#     all_paths = []
#     path_lengths = []

#     for splits, output in leaf_list:
#         outputs.append(output)
#         path_lengths.append(len(splits))

#         for side, split in splits:
#             idx = split_dict[split]

#             if side:
#                 idx += splits_length

#             all_paths.append(idx)

#     return {
#         'all_paths': all_paths,
#         'path_lengths': path_lengths,
#         'outputs': outputs,
#         'split_indices': split_indices,
#         'split_values': split_values
#     }


# class DecisionTree(tf.keras.layers.Layer):
#     def __init__(self, tree, noutputs):
#         super(DecisionTree, self).__init__()

#         self._tree = tree
#         self._noutputs = noutputs

#         # node_list = to_node_list(tree, noutputs, 0)
#         # assert len(node_list) > 1
#         # self._tensors = create_tree_tensors(node_list)
#         # self._treeify = tree_function(tree, noutputs)

#     def build(self, input_shape):
#         paths_dict = to_paths(self._tree)

#         self._path_kernel = constant(paths_dict['all_paths'], tf.int32)
#         self._path_lengths = constant(paths_dict['path_lengths'], tf.int32)
#         self._outputs = constant(paths_dict['outputs'], tf.float32)
#         self._split_indices = constant(paths_dict['split_indices'], tf.int32)
#         self._split_values = constant(paths_dict['split_values'], tf.float32)
#         # self._pathset = paths_dict['split_set']

#     def call(self, inputs):
#         nrows = tf.shape(inputs)[0]
#         np = len(self._path_lengths)

#         split_matrix = tf.gather(inputs, self._split_indices, axis=-1)
#         greater = split_matrix > self._split_values
#         less = tf.logical_not(greater)

#         split_features = tf.cast(tf.concat([less, greater], -1), tf.bool)
#         all_path_features = tf.gather(split_features, self._path_kernel, axis=-1)
#         lengths = tf.tile(self._path_lengths, (nrows,))

#         flat = tf.reshape(all_path_features, (-1,))
#         ragged = tf.RaggedTensor.from_row_lengths(flat, lengths)
#         ragged_matrix  = tf.RaggedTensor.from_uniform_row_length(ragged, np)

#         path_features = tf.reduce_all(ragged_matrix, axis=-1)
#         path_features = tf.RaggedTensor.to_tensor(path_features)
#         output_idxs = tf.where(path_features)

#         return tf.gather(self._outputs, output_idxs[:,1], 0)
#
####################################################
# Like what we ended up with, but defined recursively
####################################################
#
# def to_graph(tree, nrows, points, mask, outputs):
#     if len(tree) == 2 and tree[-1] is None:
#         all_idxs = tf.range(nrows, dtype=tf.int32)
#         out_idxs = tf.reshape(tf.boolean_mask(all_idxs, mask), [-1, 1])
#         node_output = tf.reshape(constant(tree[0]), [1, -1])
#         out_preds = tf.tile(node_output, [tf.size(out_idxs), 1])

#         outputs = tf.tensor_scatter_nd_update(outputs, out_idxs, out_preds)
#     else:
#         split_index = constant(tree[0], tf.int32)
#         split_value = constant(tree[1])

#         greater = points[:,split_index] > split_value
#         left = tf.logical_and(mask, tf.math.logical_not(greater))
#         right = tf.logical_and(mask, greater)

#         outputs = to_graph(tree[2], nrows, points, left, outputs)
#         outputs = to_graph(tree[3], nrows, points, right, outputs)

#         # out_preds = left_preds + right_preds
#         # out_idxs = left_rows + right_rows

#     return outputs

# def tree_function(tree, noutputs):
#     def get_outputs(inputs):
#         nrows = tf.shape(inputs)[0]
#         mask = tf.tile([True], (nrows,))
#         outputs = tf.zeros((nrows, noutputs), dtype=tf.float32)

#         return to_graph(tree, nrows, inputs, mask, outputs)

#         # out_preds = tf.concat(preds, 0)
#         # out_idxs = tf.concat(idxs, 1)


#     return get_outputs
