import math

from sensenet.settings

DUMMY_NETWORK = {
    'layers': [{
        'activation_function': 'softmax',
        'number_of_nodes': 2,
        'offset': 'zeros',
        'seed': 0,
        'type': 'dense',
        'weights': 'glorot_uniform'
    }],
    'preprocess': [{'index': 0, 'type': 'image'}],
    'output_exposition': {'type': 'categorical', 'values': ['a', 'b']},
    'trees': None
}

def max_pool(pool_size, strides):
    return {
        'padding': 'same',
        'pool_size': list(pool_size),
        'strides': list(strides),
        'type': 'max_pool_2d'
    }

def conv_bn_act(kernel_dimensions, nfilters, strides):
    return [{
        'bias': 'zeros',
        'kernel': 'glorot_uniform',
        'kernel_dimensions': kernel_dimensions,
        'number_of_filters': nfilters,
        'padding': 'same',
        'seed': 1,
        'strides': strides,
        'type': 'convolution_2d'
    },
    {
        'beta': 'zeros',
        'gamma': 'ones',
        'mean': 'zeros',
        'type': 'batch_normalization',
        'variance': 'ones'
    },
    {
        'activation_function': 'relu',
        'type': 'activation'
    }]

def compose_simple(input_shape):
    layers = []

    layers.extend(conv_bn_act([5, 5], 32, [4, 4]))
    layers.extend(conv_bn_act([3, 3], 32, [1, 1]))
    layers.append(max_pool([3, 3], [2, 2]))
    layers.extend(conv_bn_act([3, 3], 32, [1, 1]))
    layers.append(max_pool([3, 3], [2, 2]))
    layers.extend(conv_bn_act([3, 3], 32, [1, 1]))
    layers.append(max_pool([3, 3], [2, 2]))

    final_pixels = (input_shape[0] * input_shape[1]) / 256

    if final_pixels < 16:
        layers.append({'type': 'flatten'})
    else:
        layers.append({'type': 'global_average_pool_2d'})

    return layers

COMPOSERS = {
    'simple': compose_simple
}

def get_outputs(network):
    processor = ImageProcessor(network, Settings({}))
    print(dir(processor))
    return processor.outputs

def generate_metadata(network_name, image_layers, input_shape):
    return {
        'base_image_network': network_name,
        'input_image_shape': list(input_shape),
        'loading_method': 'centering',
        'mean_image': None,
        'outputs': None,
        'version': None
    }

def simple_network(network_name, input_shape):
    composer = COMPOSERS[network_name]

    layers = composer(input_shape)
    metadata = generate_metadata(network_name, layers, input_shape)
    image_newtork = {'layers': layers, 'metadata': metadata}

    network = dict(DUMMY_NETWORK)
    network['image_network'] = image_network
    network['image_network']['metadata']['outputs'] = get_outputs(network)

    return network
