import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import NUMERIC, CATEGORICAL, IMAGE_PATH
from sensenet.graph.layers.utils import make_tensor

def create_loaders(model, input_variables):
    Xin = input_variables['raw_X']
    preprocessors = model['preprocess']

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
        if proc['type'] == NUMERIC:
            if 'mean' in proc:
                num_idxs.append(i)
                locations.append((NUMERIC, len(means)))
                means.append(proc['mean'])

                if proc['stdev'] > 0:
                    stdevs.append(proc['stdev'])
                else:
                    # This should only happen if the feature had a constant
                    # value in training
                    stdevs.append(1)
            elif 'one_value' in proc:
                bin_idxs.append(i)
                locations.append(('binary', len(zero_values)))
                zero_values.append(proc['zero_value'])
        elif proc['type'] == CATEGORICAL:
            cat_idxs.append(i)
            locations.append((CATEGORICAL, len(depths)))
            depths.append(len(proc['values']))
        elif proc['type'] == IMAGE_PATH:
            locations.append((IMAGE_PATH, n_images))
            n_images += 1

    variables = {}

    if len(means) > 0:
        num_shape = (None, len(means))
        nX = tf.gather(Xin, num_idxs, axis=-1, name='numeric_input')
        nMean = make_tensor(means)
        nStd = make_tensor(stdevs)

        variables['numeric_out'] = (nX - nMean) / nStd

    if len(zero_values) > 0:
        bin_shape = (None, len(zero_values))
        bX = tf.gather(Xin, bin_idxs, axis=-1, name='binary_input')
        bLow = make_tensor(zero_values)

        variables['binary_out'] = tf.cast(tf.not_equal(bX, bLow), tf.float32)

    if len(depths) > 0:
        variables['categoricals'] = []

        for depth, idx in zip(depths, cat_idxs):
            cname = 'categorical_input_i'
            cX = tf.cast(Xin[:, idx], tf.int32)
            cout = tf.cast(tf.one_hot(cX, depth), tf.float32)
            variables['categoricals'].append(cout)

    return locations, variables

def reorder_inputs(locations, variables):
    to_concatenate = []

    for vtype, index in locations:
        if vtype == CATEGORICAL:
            to_concatenate.append(variables['categoricals'][index])
        elif vtype == IMAGE_PATH:
            to_concatenate.append(variables['image_out'][:,:,index])
        else:
            outvar = tf.reshape(variables[vtype + '_out'][:,index], [-1, 1])
            to_concatenate.append(outvar)

    return tf.concat(to_concatenate, -1, name='preprocessed_inputs')
