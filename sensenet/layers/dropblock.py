import sensenet.importers
tf = sensenet.importers.import_tensorflow()

class DropBlock2D(tf.keras.layers.Layer):
    def __init__(self, rate=0.5, block_size=5, seed=42, **kwargs):
        super(DropBlock2D, self).__init__(**kwargs)

        assert 0 < rate < 1

        self._seed = seed
        self._keep_prob = tf.constant(1 - rate, dtype=tf.float32)
        self._block_size = tf.constant(block_size, dtype=tf.int32)

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 4
        _, self._h, self._w, self._channel = input_shape.as_list()

        min_dim = min(self._h, self._w)
        self._block_size = min(self._block_size, min(self._h, self._w))

        # pad the mask
        p1 = (self._block_size - 1) // 2
        p0 = (self._block_size - 1) - p1

        self._padding = [[0, 0], [p0, p1], [p0, p1], [0, 0]]

        w, h, block = float(self._w), float(self._h), float(self._block_size)
        num = (1. - self._keep_prob) * (w * h) / (block ** 2)
        denom = (w - block + 1) * (h - block + 1)

        self._gamma = num / denom

        super(DropBlock2D, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        def drop():
            mask = self.create_mask(tf.shape(inputs))
            mask_size = tf.cast(tf.size(mask), dtype=tf.float32)
            masked = inputs * mask

            # Rescale inputs for stable activations
            return masked * mask_size / tf.reduce_sum(mask)

        if training is None:
            training = tf.keras.backend.learning_phase()

        is_test = tf.logical_not(training)
        return tf.cond(is_test, true_fn=lambda: inputs, false_fn=drop)

    def create_mask(self, input_shape):
        mask_shape = tf.stack([input_shape[0],
                               self._h - self._block_size + 1,
                               self._w - self._block_size + 1,
                               self._channel])

        samp = tf.random.uniform(mask_shape, dtype=tf.float32, seed=self._seed)
        bernoulli_sample = tf.nn.relu(tf.sign(self._gamma - samp))

        mask = tf.pad(bernoulli_sample, self._padding)
        block_shape = [1, self._block_size, self._block_size, 1]
        block_mask = tf.nn.max_pool(mask, block_shape, [1, 1, 1, 1], 'SAME')

        return 1 - block_mask
