import os

import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import IMAGE_STANDARDIZERS, WARP, PAD, CROP
from sensenet.accessors import get_image_shape
from sensenet.layers.utils import constant, propagate
from sensenet.layers.construct import layer_sequence

CONTRAST_LIMIT = 0.25
EXPECTED_LOW = 255 * CONTRAST_LIMIT
EXPECTED_HIGH = 255 * (1 - CONTRAST_LIMIT)

def scale_for_box(input_dims, target_dims, minimum):
    y_scale = target_dims[0] / input_dims[0]
    x_scale = target_dims[1] / input_dims[1]

    if minimum:
        return tf.math.minimum(x_scale, y_scale)
    else:
        return tf.math.maximum(x_scale, y_scale)

def resize_with_crop_or_pad(settings, target_dims, image):
    pad_only = settings.rescale_type == PAD

    # Assume target_dims are [h, w, channels]
    if len(image.shape) == 3:
        in_dims = tf.cast(tf.shape(image)[:2], tf.float32)
    elif len(image.shape) == 4:
        in_dims = tf.cast(tf.shape(image)[1:3], tf.float32)

    out_dims = tf.cast(target_dims, tf.float32)

    box_scale = scale_for_box(in_dims, out_dims, pad_only)
    scaled_dims = tf.cast(tf.math.round(in_dims * box_scale), tf.int32)
    scaled = tf.image.resize(image, scaled_dims, method='nearest')

    if pad_only:
        scaled_dims = tf.cast(scaled_dims, tf.float32)
        pad_h = [0, out_dims[0] - scaled_dims[0]]
        pad_w = [0, out_dims[1] - scaled_dims[1]]

        if len(image.shape) == 3:
            pad = [pad_h, pad_w, [0, 0]]
        elif len(image.shape) == 4:
            pad = [[0, 0], pad_h, pad_w, [0, 0]]

        return tf.pad(scaled, pad, constant_values=tf.reduce_mean(scaled))
    else:
        height, width = target_dims[0], target_dims[1]
        return tf.image.resize_with_crop_or_pad(image, height, width)

# Unused right now, but I'll leave it here just in case
def adjust_contrast(image):
    flattened = tf.sort(tf.reshape(tf.cast(image, tf.float32), (-1,)))
    npixels = tf.cast(tf.shape(flattened)[0], tf.float32)
    limit_index = tf.cast(tf.round(npixels * CONTRAST_LIMIT), tf.int32)

    pLow = flattened[limit_index]
    pHigh = flattened[-limit_index]

    low_contrast = tf.maximum(1.0, pLow / EXPECTED_LOW)
    high_contrast = tf.maximum(1.0, EXPECTED_HIGH / pHigh)
    adjustment = tf.maximum(high_contrast, low_contrast)

    return tf.image.adjust_contrast(image, adjustment)

def rescale(settings, target_shape, image):
    target_dims = tf.constant(target_shape[1:3], tf.int32)

    if settings.rescale_type is None or settings.rescale_type == WARP:
        new_image = tf.image.resize(image, target_dims, method='nearest')
    elif settings.rescale_type in [PAD, CROP]:
        new_image = resize_with_crop_or_pad(settings, target_dims, image)
    else:
        raise ValueError('Rescale type %s unknown' % settings.rescale_type)

    if image.shape[-1] == 4:
        if len(image.shape) == 4:
            new_image = new_image[:,:,:,:3]
        elif len(image.shape) == 3:
            new_image = new_image[:,:,:3]
        else:
            raise ValueError('Image tensor is rank %d' % len(image.shape))
    elif image.shape[-1] != 3:
        raise ValueError('Number of color channels is %d' % image.shape[-1])

    if settings.color_space and settings.color_space.lower().startswith('bgr'):
        return tf.reverse(new_image, axis=[-1])
    else:
        return new_image

def make_image_reader(settings, target_shape, return_original_dimensions):
    nchannels = target_shape[-1]
    input_format = settings.input_image_format or 'file'
    prefix = settings.image_path_prefix

    def read_image(path_or_bytes):
        if input_format == 'pixel_values':
            raw = path_or_bytes
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
            # `decode_image` here because the tensor that comes out
            # doesn't have a shape!
            raw = tf.io.decode_jpeg(img_bytes,
                                    dct_method='INTEGER_ACCURATE',
                                    channels=nchannels)

        if return_original_dimensions:
            return rescale(settings, target_shape, raw), tf.shape(raw)
        else:
            return rescale(settings, target_shape, raw)

    return read_image

class ImageReader(tf.keras.layers.Layer):
    def __init__(self, network, settings):
        super(ImageReader, self).__init__()

        self._input_shape = get_image_shape(network)
        self._settings = settings

    def build(self, input_shape):
        self._read = make_image_reader(self._settings, self._input_shape, False)

    def call(self, inputs):
        if self._settings.input_image_format == 'pixel_values':
            dims = tf.constant(self._input_shape[1:3], tf.int32)
            images = rescale(self._settings, self._input_shape, inputs)
        else:
            images = tf.map_fn(self._read, inputs, fn_output_signature=tf.uint8)

        return tf.cast(images, tf.float32)

class BoundingBoxImageReader(ImageReader):
    def __init__(self, network, settings):
        super(BoundingBoxImageReader, self).__init__(network, settings)

    def build(self, input_shape):
        self._read = make_image_reader(self._settings, self._input_shape, True)

    def call(self, inputs):
        if self._settings.input_image_format == 'pixel_values':
            dims = tf.constant(self._input_shape[1:3], tf.int32)
            image = rescale(self._settings, self._input_shape, inputs)
            original_dims = tf.expand_dims(tf.shape(inputs)[1:], axis=0)
        else:
            # We're explicitly saying here we only take one filename at
            # a time; if multiple files are passed in, all but the first
            # one are ignored.
            image, dims = self._read(inputs[0, 0])
            image = tf.expand_dims(image, axis=0)
            original_dims = tf.expand_dims(dims, axis=0)

        return tf.cast(image, tf.float32), original_dims

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
