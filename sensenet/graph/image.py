import os

from PIL import Image

import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import IMAGE_STANDARDIZERS, ANCHORS
from sensenet.pretrained import get_pretrained_layers
from sensenet.graph.construct import make_layers
from sensenet.graph.layers.utils import make_tensor

# def read_image(path, input_image_shape):
#     img = Image.open(path)

#     if input_image_shape:
#         in_shape = input_image_shape[:-1]

#         if input_image_shape[-1] == 1:
#             itype = 'L'
#         elif input_image_shape[-1] == 3:
#             itype = 'RGB'
#         else:
#             raise ValueError('%d is not a valid number of channels' %
#                              input_image_shape[-1])
#     else:
#         in_shape = img.size
#         itype = 'RGB'

#     img = img.convert(itype)

#     if img.size != in_shape:
#         if img.size[0] * img.size[1] > in_shape[0] * in_shape[1]:
#             img = img.resize(in_shape, Image.NEAREST)
#         else:
#             img = img.resize(in_shape, Image.BICUBIC)

#     return img

# def read_fn(image_network, image_directory):
#     input_shape = image_network['metadata']['input_image_shape']

#     def reader(image_path):
#         img = read_image(os.path.join(image_directory, image_path), input_shape)
#         X = np.array(img, dtype=np.float32)

#         if len(X.shape) == 2:
#             return np.expand_dims(X, axis=2)
#         else:
#             return X

#     return reader

def image_reader_fn(input_shape, from_file, path_prefix):
    dims = tf.constant(input_shape[:2][::-1], dtype=tf.int32)
    nchannels = input_shape[-1]

    def read_image(path_or_bytes):
        if from_file:
            if path_prefix:
                path = tf.strings.join([path_prefix, path_or_bytes])
            else:
                path = path_or_bytes

            img_bytes = tf.io.read_file(path)
        else:
            img_bytes = path_or_bytes

        raw_image = tf.io.decode_png(img_bytes, channels=nchannels)
        resized = tf.compat.v2.image.resize(raw_image, dims, method='nearest')

        return resized

    return read_image

def make_image_row_reader(input_shape, from_file, path_prefix):
    reader = image_reader_fn(input_shape, from_file, path_prefix)

    def image_row_reader(img_row):
        return tf.map_fn(reader, img_row, back_prop=False, dtype=tf.uint8)

    return image_row_reader

def image_tensor(variables, input_shape, from_file, path_prefix):
    images_in = variables['image_paths']
    row_reader = make_image_row_reader(input_shape, from_file, path_prefix)
    output = tf.map_fn(row_reader, images_in, back_prop=False, dtype=tf.uint8)

    return tf.cast(output, tf.float32)

def normalize_image(Xin, image_network):
    metadata = image_network['metadata']
    method = metadata['loading_method']
    mean, stdev = IMAGE_STANDARDIZERS[method]

    X = Xin

    if method == 'channelwise_centering':
        X = tf.reverse(X, axis=[-1])

    if metadata['mean_image'] is not None:
        mean_image = make_tensor(metadata['mean_image'])
        X = X - mean_image

    if mean != 0:
        mean_ten = make_tensor(mean)
        X = X - mean_ten

    if stdev != 1:
        stdev_ten = make_tensor(stdev)
        X = X / stdev_ten

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

def image_preprocessor(variables, image_network, from_file, path_prefix):
    network = complete_image_network(image_network)
    metadata = network['metadata']

    in_shape = metadata['input_image_shape']
    graph_shape = graph_input_shape(network)
    n_out = metadata['outputs']
    images_per_row = variables['image_paths'].shape[1]

    all_images = image_tensor(variables, in_shape, from_file, path_prefix)
    one_per_row = tf.reshape(all_images, [-1] + graph_shape[1:])
    Xin = normalize_image(one_per_row, image_network)

    _, preds = make_layers(Xin, network['layers'], None)
    outputs = tf.reshape(preds, [-1, images_per_row, n_out])

    return {'image_preds': preds, 'image_out': outputs}
