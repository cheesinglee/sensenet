import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.graph.preprocess import create_preprocessors
from sensenet.graph.preprocess import concatenate_preprocessed_inputs
from sensenet.graph.construct import make_layers

def create_dense_layers(layers_params, variables, embedded):
    if embedded:
        Xin = variables['embedded_X']
    else:
        Xin = variables['preprocessed_X']

    layers, outputs = make_layers(Xin, layers_params, None)

    return outputs

def get_output_exposition(model):
    pass

def create_topology(model):
    variables = create_placeholders()
    outex = get_output_exposition(model)

    locations, proc_vars = create_preprocessors(model['preprocess'])
    variables.update(proc_vars)

    if model.get('image_network', None):
        img_net = model['image_network']
        n_images = sum(1 for vtype, _ in locations if vtype == 'image')
        img_vars = image_preprocessor(img_net, n_images, variables)
        variables.update(img_vars)

    preprocessed = concatenate_preprocessed_inputs(locations, variables)
    variables['preprocessed_X'] = preprocessed

    if model.get('trees', None):
        trees = model['trees']
        tree_vars = forest_preprocessor(tree_vars, variables)
        variables.update(tree_vars)

    if 'networks' in model:
        raise ValueError('No network searches yet!')
    else:
        pass
