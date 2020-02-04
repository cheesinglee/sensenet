import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import IMAGE_STANDARDIZERS
from sensenet.pretrained import complete_image_network
from sensenet.graph.layers.utils import constant

class ImagePreprocessor(tf.keras.layers.Layer):
    def __init__(self, image_network, extras):
        super(ImagePreprocessor, self).__init__()

        network = complete_image_network(image_network)
        metadata = network['metadata']

        ishape = metadata['input_image_shape']
        self._input_shape = [None, ishape[1], ishape[0], ishape[2]]

        method = metadata['loading_method']
        mimg = metadata['mean_image']
        mean, std = IMAGE_STANDARDIZERS[method]

        self._reverse = method == 'channelwise_standardizing'
        self._mean = constant(mean) if mean != 0 else None
        self._stdev = constant(std) if std != 1 else None
        self._mean_image = constant(mimg) if mimg is not None else None

        self._image_layers = make_layers(network['layers'])

        if extras:
            self._path_prefix = extras.get('path_prefix', None)
            self._input_format = extras.get('input_image_format', 'file')

    def read_fn(self):
        dims = tf.constant(self._input_shape[:2][::-1], tf.int32)
        nchannels = self._input_shape[-1]

        def read_image(path_or_bytes):
            if input_format == 'file':
                path = path_or_bytes

                if self._path_prefix:
                    path = tf.strings.join([self._path_prefix, path])

                img_bytes = tf.io.read_file(path)
            else:
                img_bytes = path_or_bytes

            raw_image = tf.io.decode_png(img_bytes, channels=nchannels)

            return tf.image.resize(raw_image, dims, method='nearest')

        return read_image

    def load_inputs(self, inputs):
        images = tf.map_fn(self._read, inputs, back_prop=False, dtype=tf.uint8)
        images = tf.cast(images, tf.float32)

        if self._reverse:
            images = tf.reverse(images, axis=[-1])

        if self._mean_image is not None:
            images = images - self._mean_image

        if self._mean is not None:
            images = images - self._mean

        if self._stdev is not None:
            images = images / self._stdev

        return images

    def build(self, input_shape):
        self._read = self.read_fn()

    def call(self, inputs):
        next_inputs = self.load_inputs(inputs)

        for layer in self._image_layers:
            next_inputs = layer(next_inputs)

        return next_inputs
