import os

import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import IMAGE_STANDARDIZERS
from sensenet.accessors import get_image_shape
from sensenet.layers.utils import constant, propagate
from sensenet.layers.construct import layer_sequence

def resize_and_pad(image, dims):
    img_shape = tf.shape(image)[:2].numpy()
    y_scale = float(dims[0]) / img_shape[0]
    x_scale = float(dims[1]) / img_shape[1]
    out_shape = [int(round(d * min(x_scale, y_scale))) for d in img_shape]

    assert dims[0] == out_shape[0] or dims[1] == out_shape[1]

    pad = [[0, d - os] for d, os in zip(dims, out_shape)] + [[0, 0]]
    img = tf.image.resize(image, out_shape, method='nearest')

    return tf.pad(img, pad)

def get_image_reader_fn(image_shape, input_format, prefix):
    dims = tf.constant(image_shape[1:3], tf.int32)
    nchannels = image_shape[-1]

    def read_image(path_or_bytes):
        if input_format == 'pixel_values':
            raw_image = path_or_bytes
        else:
            if input_format == 'file':
                path = path_or_bytes

                if prefix:
                    path = tf.strings.join([prefix + os.sep, path])

                img_bytes = tf.io.read_file(path)
            else:
                img_bytes = path_or_bytes

            # Note that, spectacularly weirdly, this method will also
            # work for pngs and gifs.  Even wierder, We can't use
            # decode_image here because the tensor that comes out
            # doesn't have a shape!
            raw_image = tf.io.decode_jpeg(img_bytes,
                                          dct_method='INTEGER_ACCURATE',
                                          channels=nchannels)

        return tf.image.resize(raw_image, dims, method='nearest')

    return read_image

class ImageReader(tf.keras.layers.Layer):
    def __init__(self, network, settings):
        super(ImageReader, self).__init__()

        self._input_shape = get_image_shape(network)
        self._path_prefix = settings.image_path_prefix
        self._input_format = settings.input_image_format or 'file'

    def build(self, input_shape):
        if self._input_format != 'pixel_values':
            self._read = get_image_reader_fn(self._input_shape,
                                             self._input_format,
                                             self._path_prefix)

    def call(self, inputs):
        if self._input_format == 'pixel_values':
            dims = tf.constant(self._input_shape[1:3], tf.int32)
            images = tf.image.resize(inputs, dims, method='nearest')
        else:
            images = tf.map_fn(self._read, inputs, back_prop=False, dtype=tf.uint8)

        return tf.cast(images, tf.float32)

class ImageLoader(tf.keras.layers.Layer):
    def __init__(self, network):
        super(ImageLoader, self).__init__()

        metadata = network['metadata']
        method = metadata['loading_method']
        mimg = metadata['mean_image']
        mean, std = IMAGE_STANDARDIZERS[method]

        self._reverse = method == 'channelwise_centering'
        self._mean = constant(mean) if mean != 0 else None
        self._stdev = constant(std) if std != 1 else None
        self._mean_image = constant(mimg) if mimg is not None else None

    def call(self, inputs):
        images = inputs

        if self._reverse:
            images = tf.reverse(images, axis=[-1])

        if self._mean_image is not None:
            images = images - self._mean_image

        if self._mean is not None:
            images = images - self._mean

        if self._stdev is not None:
            images = images / self._stdev

        return images

class ImagePreprocessor(tf.keras.layers.Layer):
    def __init__(self, image_network, settings):
        super(ImagePreprocessor, self).__init__()

        self._reader = ImageReader(image_network, settings)
        self._loader = ImageLoader(image_network)
        self._image_layers = layer_sequence(image_network)

    def call(self, inputs):
        raw_images = self._reader(inputs)
        images = self._loader(raw_images)

        return propagate(self._image_layers, images)
