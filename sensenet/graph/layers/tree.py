import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.graph.layers.utils import make_tensor

def create_tree_variables(node_list):
    assert [n['node_id'] for n in node_list] == list(range(len(node_list)))

    tree_vars = {}

    for key in ['split_index', 'split_value', 'outputs']:
        tree_vars[key] = [node[key] for node in node_list]

    tree_vars['next_matrix'] = [[n['left'], n['right']] for n in node_list]
    tree_vars['is_leaf'] = [n['left'] == n['node_id'] for n in node_list]

    for k in tree_vars:
        assert len(tree_vars[k]) == len(node_list)

    return tree_vars

def build_tree_topology(Xin, node_list):
    variables = create_tree_variables(node_list)

    is_leaf = make_tensor(variables['is_leaf'], ttype=tf.bool)
    indexes = make_tensor(variables['split_index'], ttype=tf.int32)
    split_values = make_tensor(variables['split_value'], ttype=tf.float32)
    next_matrix = make_tensor(variables['next_matrix'], ttype=tf.int32)
    outputs = make_tensor(variables['outputs'], ttype=tf.float32)

    nrows = tf.shape(Xin)[0]
    nouts = len(variables['outputs'][0])

    xcoords = tf.range(nrows, dtype=tf.int32)
    zero_idxs = tf.zeros([nrows, 1], dtype=tf.int32)
    first_output = tf.zeros([nrows, nouts], dtype=tf.float32)

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
        next_outputs = tf.gather(outputs, next_nodes)

        return tf.reshape(next_nodes, [-1, 1]), tf.reshape(next_outputs, [-1, 3])

    _, preds = tf.while_loop(loop_cond, loop_body, [zero_idxs, first_output])

    return preds

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
