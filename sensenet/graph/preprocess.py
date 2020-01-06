import sensenet.importers
tf = sensenet.importers.import_tensorflow()

def create_preprocessors(preprocessors):
    locations = []

    num_idxs = []
    means = []
    stdevs = []

    bin_idxs = []
    zero_values = []

    cat_idxs = []
    depths = []
    n_images = 0

    for i, proc in enumerate(preprocessors):
        if proc['type'] == 'numeric':
            if 'mean' in proc:
                num_idxs.append(i)
                locations.append(('numeric', len(means)))
                means.append(proc['mean'])
                stdevs.append(proc['stdev'])
            elif 'one_value' in proc:
                bin_idxs.append(i)
                locations.append(('binary', len(zero_values)))
                zero_values.append(proc['zero_value'])
        elif proc['type'] == 'categorical':
            cat_idxs.append(i)
            locations.append(('categorical', len(depths)))
            depths.append(len(proc['values']))
        elif proc['type'] == 'image':
            locations.append(('image', n_images))
            n_images += 1

    variables = {}

    if len(means) > 0:
        num_shape = (None, len(means))
        nX = tf.gather(tf.float32, shape=num_shape, name='numeric_input')
        nMean = tf.constant(means, dtype=tf.float32)
        nStd = tf.constant(stdevs, dtype=tf.float32)

        variables['numeric_X'] = nX
        variables['numeric_out'] = (nX - nMean) / nStd

    if len(zero_values) > 0:
        bin_shape = (None, len(zero_values))
        bX = tf.placeholder(tf.float32, shape=bin_shape, name='binary_input')
        bLow = tf.constant(zero_values, dtype=tf.float32)

        variables['binary_X'] = bX
        variables['binary_out'] = bX != bLow

    if len(depths) > 0:
        for depth in depths:
            cname = 'categorical_input_i'
            cX = tf.placeholder(tf.float32, shape=(None,), name=cname)
            cout = tf.one_hot(cX, depth)
            variables['categoricals'].append({
                'input': cX,
                'output': cout
            })

    return locations, variables

def concatenate_preprocessed_inputs(locations, variables):
    to_concatenate = []

    for vtype, index in locations:
        if vtype == 'categorical':
            to_concatenate.append(variables['categoricals'][index]['output'])
        elif vtype == 'image':
            to_concatenate.append(variables['image_out'][:,:,index])
        else:
            outname = vtype + '_out'
            to_concatenate.append(variables[outname][:,index])

    return tf.concat(to_concatenate, 0, name='preprocessed_inputs')
