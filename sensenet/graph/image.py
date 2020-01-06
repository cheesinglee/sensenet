from PIL import Image

import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import IMAGE_STANDARDIZERS, ANCHORS
from sensenet.pretrained import get_pretrained_layers
from sensenet.graph.construct import make_layers
from sensenet.graph.layers.utils import make_tensor

def read_image(path, input_image_shape):
    img = Image.open(path)

    if input_image_shape:
        in_shape = input_image_shape[:-1]

        if input_image_shape[-1] == 1:
            itype = 'L'
        elif input_image_shape[-1] == 3:
            itype = 'RGB'
        else:
            raise ValueError('%d is not a valid number of channels' %
                             input_image_shape[-1])
    else:
        in_shape = img.size
        itype = 'RGB'

    img = img.convert(itype)

    if img.size != in_shape:
        if img.size[0] * img.size[1] > in_shape[0] * in_shape[1]:
            img = img.resize(in_shape, Image.NEAREST)
        else:
            img = img.resize(in_shape, Image.BICUBIC)

    return img

def read_fn(image_network):
    input_shape = image_network['metadata']['input_image_shape']

    def reader(image_path):
        img = read_image(image_path, input_shape)
        X = np.array(img, dtype=np.float32)

        if len(X.shape) == 2:
            return np.expand_dims(X, axis=2)
        else:
            return X

    return reader

def normalize_image(Xin, image_network):
    metadata = image_network['metadata']
    method = metadata['loading_method']
    mean, stdev = IMAGE_STANDARDIZERS[method]

    X = Xin

    if method == 'channelwise_centering':
        X = tf.reverse(X, axis=[-1])

    if mean != 0:
        mean_ten = make_tensor(mean)
        X = X - mean_ten

    if stdev != 1:
        stdev_ten = make_tensor(stdev)
        X = X / stdev_ten

    if metadata['mean_image'] is not None:
        mean_image = make_tensor(metadata['mean_image'])
        X = X - mean_image

    return X

def complete_image_network(network, top_layers=None):
    if network['layers'] is None:
        network['layers'] = get_pretrained_layers(network)
        metadata = network['metadata']

        assert metadata.get('mean_image', None) is None
        network['metadata']['mean_image'] = None

        if 'output_indices' in metadata:
            if metadata.get('anchors', None) is None:
                anchors = ANCHORS[metadata['base_image_network']]
                network['metadata']['anchors'] = anchors

    if top_layers:
        network['layers'] += top_layers

    return network

def graph_input_shape(image_network):
    input_shape = image_network['metadata']['input_image_shape']
    assert len(input_shape) == 3 and input_shape[-1] in [1, 3]
    return [None, input_shape[1], input_shape[0], input_shape[2]]

def image_preprocessor(image_network, images_per_row):
    network = complete_image_network(image_network)
    metadata = network['metadata']

    in_shape = graph_input_shape(network)
    all_shape = [None, images_per_row] + in_shape[1:]
    n_out = metadata['outputs']

    X = tf.placeholder(tf.float32, shape=all_shape, name='image_input')
    all_images = tf.reshape(X, [-1] + in_shape[1:])
    Xin = normalize_image(all_images, image_network)

    _, preds = make_layers(Xin, network['layers'], None)
    outputs = tf.reshape(preds, [-1, images_per_row, n_out])

    return {'image_X': X, 'image_preds': preds, 'image_out': outputs}
