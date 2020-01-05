import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.graph.preprocess import create_preprocessors
from sensenet.graph.preprocess import concatenate_preprocessed_inputs
from sensenet.graph.construct import make_layers

def create_placeholders():
    return {
        'is_training': tf.placeholder(tf.bool, name='is_training'),
        'keep_prob': tf.placeholder(tf.float32, name='keep_prob')
    }

def create_dense_layers(Xin, layers_params, variables):
    trn = variables['is_training']
    kp = variables['keep_prob']

    layers, outputs = make_layers(Xin, trn, layers_params, keep_prob=kp)

    return outputs

def create_topology(model):
    variables = create_placeholders()
    locations, proc_vars = create_preprocessors(model['preprocess'])

    if model.get('image_network', None):
        img_net = model['image_network']
        n_images = sum(1 for vtype, _ in locations if vtype == 'image')
        img_vars = image_preprocessor(img_net, n_images, variables)

    variables.update(proc_vars)
    variables.update(img_vars)

    dense_input = concatenate_preprocessed_inputs(locations, variables)
