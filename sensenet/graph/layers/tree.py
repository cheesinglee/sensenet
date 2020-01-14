import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.accessors import number_of_outputs
from sensenet.graph.layers.utils import make_tensor

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

def build_tree_topology(Xin, node_list):
    assert len(node_list) > 1
    tensors = create_tree_tensors(node_list)

    is_leaf = make_tensor(tensors['is_leaf'], ttype=tf.bool)
    indexes = make_tensor(tensors['split_index'], ttype=tf.int32)
    split_values = make_tensor(tensors['split_value'], ttype=tf.float32)
    next_matrix = make_tensor(tensors['next_matrix'], ttype=tf.int32)
    outputs = make_tensor(tensors['outputs'], ttype=tf.float32)

    nrows = tf.shape(Xin)[0]
    nouts = len(tensors['outputs'][0])

    xcoords = tf.range(nrows, dtype=tf.int32)
    zero_idxs = tf.zeros([nrows, 1], dtype=tf.int32)
    first_output = tf.ones([nrows, nouts], dtype=tf.float32)

    def loop_cond(current_nodes, _):
        gathered = tf.gather(is_leaf, current_nodes)
        reduced = tf.reduce_all(gathered)
        return tf.logical_not(reduced)

    def loop_body(current_nodes, _):
        sidxs = tf.squeeze(tf.gather(indexes, current_nodes))
        svals = tf.squeeze(tf.gather(split_values, current_nodes))

        value_coords = tf.stack([xcoords, sidxs], axis=1)
        values = tf.gather_nd(Xin, value_coords)
        side = tf.dtypes.cast(values > svals, tf.int32)
        node_coords = tf.stack([tf.squeeze(current_nodes), side], axis=1)

        next_nodes = tf.gather_nd(next_matrix, node_coords)
        next_outputs = tf.reshape(tf.gather(outputs, next_nodes), [-1, nouts])

        return tf.reshape(next_nodes, [-1, 1]), next_outputs

    _, preds = tf.while_loop(loop_cond, loop_body, [zero_idxs, first_output])

    return preds

def nodes_to_tensor(Xin, node_list):
    if len(node_list) > 1:
        return build_tree_topology(Xin, node_list)
    else:
        nrows = tf.shape(Xin)[0]
        outten = tf.reshape(make_tensor(node_list[0]['outputs']), [1, -1])

        return tf.tile(outten, [nrows, 1])

def to_forest(Xin, node_lists, normalize):
    all_preds = []

    for nlist in node_lists:
        all_preds.append(nodes_to_tensor(Xin, nlist))

    summed = tf.add_n(all_preds)
    aplen = make_tensor(len(all_preds))

    if normalize == 'mean':
        return summed / aplen
    else:
        raise ValueError('Normalizer "%s" unknown' % normalize)

def forest_preprocessor(model, variables):
    all_preds = []

    Xin = variables['preprocessed_X']
    noutputs = number_of_outputs(model)

    for input_range, trees in model['trees']:
        start, end = input_range
        fXin = Xin[:,start:end]
        node_lists = [to_node_list(t, noutputs, 0) for t in trees]
        aggregated = to_forest(fXin, node_lists, 'mean')

        all_preds.append(aggregated)

    tree_X = tf.concat(all_preds, -1)
    embedded_X = tf.concat([tree_X, Xin], -1)

    return embedded_X
