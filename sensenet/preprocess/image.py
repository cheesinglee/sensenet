import os

import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import IMAGE_STANDARDIZERS
from sensenet.accessors import get_image_shape
from sensenet.layers.utils import constant, propagate
from sensenet.layers.construct import layer_sequence

def scale_for_box(input_dims, target_dims):
    y_scale = target_dims[0] / input_dims[0]
    x_scale = target_dims[1] / input_dims[1]

    return tf.math.minimum(x_scale, y_scale)

def resize_and_pad(image, input_dims, target_dims):
    # Assume both input_dims and target_dims are [h, w, channels]
    img_shape = tf.cast(input_dims, tf.float32)
    box_scale = scale_for_box(img_shape, tf.cast(target_dims, tf.float32))
    out_shape = tf.cast(tf.math.round(img_shape * box_scale), tf.int32)

    pad_h = [0, target_dims[0] - out_shape[0]]
    pad_w = [0, target_dims[1] - out_shape[1]]

    if len(image.shape) == 3:
        pad = [pad_h, pad_w, [0, 0]]
    elif len(image.shape) == 4:
        pad = [[0, 0], pad_h, pad_w, [0, 0]]

    img = tf.image.resize(image, out_shape, method='nearest')

    return tf.pad(img, pad)

def get_image_reader_fn(image_shape, input_format, prefix, pad=False):
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

        if pad:
            img_shape = tf.shape(raw_image)
            return resize_and_pad(raw_image, img_shape[:2], dims), img_shape
        else:
            return tf.image.resize(raw_image, dims, method='nearest')

    return read_image

class ImageReader(tf.keras.layers.Layer):
    def __init__(self, network, settings):
        super(ImageReader, self).__init__()

        self._input_shape = get_image_shape(network)
        self._path_prefix = settings.image_path_prefix
        self._input_format = settings.input_image_format or 'file'

    def make_reader(self, do_padding):
        return get_image_reader_fn(self._input_shape,
                                   self._input_format,
                                   self._path_prefix,
                                   pad=do_padding)

    def build(self, input_shape):
        if self._input_format != 'pixel_values':
            self._read = self.make_reader(False)

    def call(self, inputs):
        if self._input_format == 'pixel_values':
            dims = tf.constant(self._input_shape[1:3], tf.int32)
            images = tf.image.resize(inputs, dims, method='nearest')
        else:
            images = tf.map_fn(self._read, inputs, fn_output_signature=tf.uint8)

        return tf.cast(images, tf.float32)


class BoundingBoxImageReader(ImageReader):
    def __init__(self, network, settings):
        super(BoundingBoxImageReader, self).__init__(network, settings)

    def build(self, input_shape):
        if self._input_format != 'pixel_values':
            self._read = self.make_reader(True)

    def call(self, inputs):
        if self._input_format == 'pixel_values':
            original_dims = tf.expand_dims(tf.shape(inputs)[1:], axis=0)
            dims = tf.constant(self._input_shape[1:3], tf.int32)
            images = resize_and_pad(inputs, tf.shape(inputs)[1:3], dims)

            outputs = images, tf.tile(original_dims, [tf.shape(inputs)[0], 1])
        else:
            outsig = (tf.uint8, tf.int32)
            outputs = tf.map_fn(self._read, inputs, fn_output_signature=outsig)

        return tf.cast(outputs[0], tf.float32), outputs[1]

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
