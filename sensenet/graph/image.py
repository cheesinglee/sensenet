from PIL import Image

import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import CAFFE_MEAN, TORCH_MEAN, TORCH_STD, ANCHORS
from sensenet.pretrained import get_pretrained_layers
from sensenet.graph.construct import make_layers

def raw(X):
    return X

def centering(X):
    return X / 127.5 - 1.

def skewed_centering(X):
    return X / 128. - 1.

def channelwise_centering(X):
    Xout = X[..., ::-1]

    Xout[..., 0] -= CAFFE_MEAN[0]
    Xout[..., 1] -= CAFFE_MEAN[1]
    Xout[..., 2] -= CAFFE_MEAN[2]

    return Xout

def normalizing(X):
    return X / 255.

def channelwise_standardizing(X):
    Xout = X / 255.

    Xout[..., 0] -= TORCH_MEAN[0]
    Xout[..., 1] -= TORCH_MEAN[1]
    Xout[..., 2] -= TORCH_MEAN[2]

    Xout[..., 0] /= TORCH_STD[0]
    Xout[..., 1] /= TORCH_STD[1]
    Xout[..., 2] /= TORCH_STD[2]

    return Xout

IMAGE_NORMALIZERS = {
    None: raw,
    'normalizing': normalizing,
    'centering': centering,
    'skewed_centering': skewed_centering,
    'channelwise_centering': channelwise_centering,
    'channelwise_standardizing': channelwise_standardizing
}

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

def make_loader(network):
    metadata = network['metadata']
    input_shape = metadata['input_image_shape']
    normalize_fn = IMAGE_NORMALIZERS[metadata['loading_method']]

    def load(path):
        img = read_image(path, input_shape)
        X = np.array(img, dtype=np.float32)

        if len(X.shape) == 2:
            Xout = np.expand_dims(X, axis=2)
        else:
            Xout = X

        if normalize_fn is None:
            return np.expand_dims(Xout, axis=0)
        else:
            return normalize_fn(np.expand_dims(Xout, axis=0))

    return load

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
    return (None, input_shape[1], input_shape[0], input_shape[2])

def image_preprocessor(image_network, images_per_row, variables):
    network = complete_image_network(image_network)
    metadata = network['metadata']

    in_shape = graph_input_shape(network)
    n_out = metadata['outputs']

    X = tf.placeholder(tf.float32, shape=in_shape, name='image_input')
    is_training = variables['is_training']
    keep_prob = variables['keep_prob']

    if metadata['mean_image'] is not None:
        mean_image = tf.constant(metadata['mean_image'], dtype=tf.float32)
        Xin = X - mean_image
    else:
        Xin = X

    _, preds = make_layers(Xin, is_training, network['layers'], keep_prob)
    outputs = tf.reshape(preds, [-1, n_out, images_per_row])

    return {'image_X': X, 'image_preds': preds, 'image_out': outputs}
