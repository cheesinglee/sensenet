import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import IMAGE_PATH
from sensenet.graph.preprocess import create_preprocessors
from sensenet.graph.preprocess import concatenate_preprocessed_inputs
from sensenet.graph.construct import make_layers

def create_dense_layers(layers_params, variables, embedded):
    if embedded:
        Xin = variables['embedded_X']
    else:
        Xin = variables['preprocessed_X']

    _, outputs = make_layers(Xin, layers_params, None)

    return outputs

def get_output_exposition(model):
    pass

def create_topology(model):
    outex = get_output_exposition(model)
    preprocessors = model['preprocess']

    Xin = tf.placeholder(tf.float32, (None, len(preprocessors)))
    locations, variables = create_preprocessors(Xin, model['preprocess'])

    if model.get('image_network', None):
        img_net = model['image_network']
        n_images = sum(1 for vtype, _ in locations if vtype == IMAGE_PATH)
        img_vars = image_preprocessor(img_net, n_images)
        variables.update(img_vars)

    preprocessed = concatenate_preprocessed_inputs(locations, variables)
    variables['preprocessed_X'] = preprocessed

    trees = model.get('trees', None)

    if trees:
        tree_vars = forest_preprocessor(tree_vars, variables)
        variables.update(tree_vars)

    if 'networks' in model:
        raise ValueError('No network searches yet!')
    else:
        outputs = create_dense_layers(model['layers'], variables, trees)

    return Xin, outputs
