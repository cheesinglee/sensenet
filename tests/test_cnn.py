import json
import warnings

import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from keras.models import Sequential
from keras.layers import ZeroPadding2D, MaxPooling2D, AveragePooling2D, Dense
from keras.layers import BatchNormalization, Conv2D, Flatten, Activation
from keras.layers import SeparableConv2D, DepthwiseConv2D, LeakyReLU
from keras.layers import UpSampling2D, Concatenate
from keras.initializers import RandomUniform
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

from sensenet.constants import LEAKY_RELU_ALPHA

from sensenet.graph.construct import LAYER_FUNCTIONS
from sensenet.graph.keras.create import construct, constructed_metadata
from sensenet.graph.keras.extract import extract

VS = 'VarianceScaling'
RU = 'RandomUniform'

def layer_outputs(data, layer_type, keras_layer, normalizing=False):
    tfX = tf.constant(data)
    params = extract(keras_layer)

    if normalizing:
        trn = tf.constant(False)
        _, outputs = LAYER_FUNCTIONS[layer_type](tfX, params, trn)
    else:
        _, outputs = LAYER_FUNCTIONS[layer_type](tfX, params)

    return outputs

def test_padding():
    lType = "padding_2d"

    X_base = np.array([[1, 2, 3, 4], [4, 5, 6, 8], [7, 8, 9, 10]])
    i1 = np.array([X_base - 1, X_base, X_base + 1]).reshape([3, 4, 3])
    i2 = np.array([X_base - 10, X_base, X_base + 10]).reshape([3, 4, 3])

    data = np.array([i1, i2])

    with tf.Session() as sess:
        for pd in [(3, 3), (2, 3), (1, 3), (3, 0), (0, 0)]:
            model = Sequential()
            model.add(ZeroPadding2D(padding=pd, input_shape=(3, 4, 3)))

            Xp = model.predict(data)
            outputs = layer_outputs(data, lType, model.layers[0])

            assert Xp.shape == (2, 3 + pd[0] * 2, 4 + pd[1] * 2, 3), Xp.shape

            tfp = outputs.eval()

            assert Xp.tolist() == tfp.tolist(), (tfp.shape, Xp.shape)

def test_pool():
    X_base = np.array([[1, 2, 3, 4], [4, 5, 6, 8], [7, 8, 9, 10]])
    i1 = np.array([X_base - 1, X_base, X_base + 1]).reshape([3, 4, 3])
    i2 = np.array([X_base - 10, X_base, X_base + 10]).reshape([3, 4, 3])

    data = np.array([i1, i2], dtype=np.float32)

    with tf.Session() as sess:
        for ps in [(2, 2), (3, 3), (1, 3)]:
            for st in [(1, 1), (2, 2), (2, 1)]:
                for pd in ["valid", "same"]:
                    for fns in [(MaxPooling2D, "max_pool_2d"),
                                (AveragePooling2D, "average_pool_2d")]:

                        kLayer, tLayer = fns
                        model = Sequential()
                        model.add(kLayer(ps,
                                         strides=st,
                                         padding=pd,
                                         input_shape=(3, 4, 3)))

                        Xp = model.predict(data)
                        outputs = layer_outputs(data, tLayer, model.layers[0])
                        tfp = outputs.eval()

                        assert Xp.tolist() == tfp.tolist(), (tfp - Xp)

def test_batchnorm():
    lType = "batch_normalization"

    X_base = np.array([[1, 2], [4, 5], [7, 8]])
    i1 = np.array([X_base * -11, X_base, X_base + 0.2]).reshape([3, 2, 3])
    i2 = np.array([X_base - 10, X_base, X_base * 10]).reshape([3, 2, 3])

    data = np.array([i1, i2], dtype=np.float32)

    i1 = RandomUniform(minval=0.01, maxval=100)
    i2 = RandomUniform(minval=100, maxval=1000)
    i3 = RandomUniform(minval=1, maxval=1)

    with tf.Session() as sess:
        for init in [i1, i2, i3]:
            model = Sequential()
            model.add(BatchNormalization(axis=3,
                                         beta_initializer=init,
                                         gamma_initializer=init,
                                         moving_mean_initializer=init,
                                         moving_variance_initializer=init,
                                         input_shape=(3, 2, 3)))

            Xp = model.predict(data)
            outputs = layer_outputs(data, lType, model.layers[0], True)

            sess.run(tf.global_variables_initializer())
            tfp = outputs.eval()

            assert tfp.shape == Xp.shape, (tfp.shape, Xp.shape)
            assert np.allclose(Xp, tfp, atol=1e-8), (tfp - Xp)

def test_convolution():
    lType = "convolution_2d"

    X_base = np.array([[1, 2, 3, 4], [4, 5, 6, 8], [7, 8, 9, 10]])
    i1 = np.array([X_base / -11.0, X_base, X_base + 1.1]).reshape([3, 4, 3])
    i2 = np.array([X_base / 10.0, X_base, X_base * 3]).reshape([3, 4, 3])

    data = np.array([i1, i2], dtype=np.float32)
    filters = 8

    with tf.Session() as sess:
        for ksize in [(1, 1), (3, 3), (1, 3)]:
            for st in [(1, 1), (2, 2), (2, 1)]:
                for pd in ["valid", "same"]:
                    model = Sequential()
                    model.add(Conv2D(filters,
                                     ksize,
                                     strides=st,
                                     padding=pd,
                                     kernel_initializer=VS,
                                     bias_initializer=RU,
                                     input_shape=(3, 4, 3)))

                    Xp = model.predict(data)
                    outputs = layer_outputs(data, lType, model.layers[0])

                    sess.run(tf.global_variables_initializer())
                    tfp = outputs.eval()

                    assert Xp.tolist() == tfp.tolist(), (tfp - Xp)

def test_upsample():
    lType = "upsampling_2d"

    X_base = np.array([[1, 2, 3, 4], [4, 5, 6, 8], [7, 8, 9, 10]])
    i1 = np.array([X_base / -11.0, X_base, X_base + 1.1]).reshape([3, 4, 3])
    i2 = np.array([X_base / 10.0, X_base, X_base * 3]).reshape([3, 4, 3])

    data = np.array([i1, i2], dtype=np.float32)
    filters = 8

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for size in [2, (2, 2), (3, 3)]:
            model = Sequential()
            model.add(UpSampling2D(size, input_shape=(3, 4, 3)))

            Xp = model.predict(data)
            outputs = layer_outputs(data, lType, model.layers[0])
            tfp = outputs.eval()

            assert Xp.tolist() == tfp.tolist(), (tfp - Xp)

def test_flatten():
    lType = "flatten"

    X_base = np.array([[1, 2, 3, 4], [4, 5, 6, 8], [2, 8, 9, 10]])
    i1 = np.array([X_base / -11.0, X_base, X_base + 1.1]).reshape([3, 4, 3])
    i2 = np.array([X_base / 10.0, X_base, X_base * 3]).reshape([3, 4, 3])

    data = np.array([i1, i2], dtype=np.float32)

    model = Sequential()
    model.add(Flatten(input_shape=(3, 4, 3)))

    Xp = model.predict(data)
    outputs = layer_outputs(data, lType, model.layers[0])

    assert Xp.shape == (2, 36)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tfp = outputs.eval()

    assert Xp.tolist() == tfp.tolist(), (tfp - Xp)

def test_activate():
    lType = "activation"

    X_base = np.array([[1, 2e5, 3, 4], [4, 2, 6, 8e-5], [2, 8, 9, 10]])
    i1 = np.array([X_base / -0.1, X_base, X_base * 2e5]).reshape([3, 4, 3])
    i2 = np.array([X_base * -3.3e5, X_base, X_base]).reshape([3, 4, 3])

    data = np.array([i1, i2], dtype=np.float32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for act_fn in ['relu', 'tanh', 'softplus', 'sigmoid', 'linear',
                       'softmax', 'selu', 'leaky_relu']:

            model = Sequential()

            if act_fn == 'leaky_relu':
                model.add(LeakyReLU(alpha=LEAKY_RELU_ALPHA,
                                    input_shape=(3, 4, 3)))
            else:
                model.add(Activation(act_fn, input_shape=(3, 4, 3)))

            Xp = model.predict(data)
            outputs = layer_outputs(data, lType, model.layers[0])

            tfp = outputs.eval()

            assert np.allclose(Xp, tfp), (tfp - Xp)

def test_dense():
    lType = "dense"

    i1 = np.array([1, -2, 3, -4, 5, -6, 7, -8]) * 1e5
    i2 = np.array([1, -2, 3, -4, 5, -6, 7, -8]) * 1e-5

    data = np.array([i1, i2], dtype=np.float32)

    with tf.Session() as sess:
        for act_fn in ['relu', 'tanh', 'softplus', 'sigmoid', 'softmax', None]:
            model = Sequential()
            model.add(Dense(32,
                            activation=act_fn,
                            kernel_initializer='glorot_uniform',
                            bias_initializer='RandomUniform',
                            input_shape=(len(i1),)))

            Xp = model.predict(data)
            outputs = layer_outputs(data, lType, model.layers[0])

            assert Xp.shape == (2, 32)

            sess.run(tf.global_variables_initializer())
            tfp = outputs.eval()

            assert tfp.shape == Xp.shape, (tfp.shape, Xp.shape)
            assert np.allclose(Xp, tfp, atol=1e-8), (tfp - Xp)

def test_separable_convolution():
    lType = "separable_convolution_2d"

    X_base = np.array([[1, 2, -3, 4], [4, 5, -6, 8], [7, 8, 9, -10]])
    i1 = np.array([X_base / -11.0, X_base, X_base + 1.1]).reshape([3, 4, 3])
    i2 = np.array([X_base / 10.0, X_base, X_base * 3]).reshape([3, 4, 3])

    data = np.array([i1, i2], dtype=np.float32)
    filters = 8

    with tf.Session() as sess:
        for ksize in [(1, 1), (3, 3), (1, 3)]:
            for st in [(1, 1), (2, 2)]:
                for pd in ["valid", "same"]:
                    for bias in [True, False]:
                        model = Sequential()
                        model.add(SeparableConv2D(filters,
                                                  ksize,
                                                  use_bias=bias,
                                                  strides=st,
                                                  padding=pd,
                                                  kernel_initializer=VS,
                                                  bias_initializer=RU,
                                                  input_shape=(3, 4, 3)))

                        Xp = model.predict(data)
                        outputs = layer_outputs(data, lType, model.layers[0])

                        sess.run(tf.global_variables_initializer())
                        tfp = outputs.eval()

                        assert Xp.tolist() == tfp.tolist(), (tfp - Xp)

def test_depthwise_convolution():
    lType = "depthwise_convolution_2d"

    X_base = np.array([[1, 2, -3, 4], [4, 5, -6, 8], [7, 8, 9, -10]])
    i1 = np.array([X_base / -11.0, X_base, X_base + 1.1]).reshape([3, 4, 3])
    i2 = np.array([X_base / 10.0, X_base, X_base * 3]).reshape([3, 4, 3])

    data = np.array([i1, i2], dtype=np.float32)
    filters = 4

    with tf.Session() as sess:
        for ksize in [(1, 1), (3, 3), (1, 3)]:
            for st in [(1, 1), (2, 2)]:
                for pd in ["valid", "same"]:
                    for bias in [True, False]:
                        model = Sequential()
                        model.add(DepthwiseConv2D(kernel_size=ksize,
                                                  use_bias=bias,
                                                  strides=st,
                                                  padding=pd,
                                                  kernel_initializer=VS,
                                                  bias_initializer=RU,
                                                  input_shape=(3, 4, 3)))

                        Xp = model.predict(data)
                        outputs = layer_outputs(data, lType, model.layers[0])

                        sess.run(tf.global_variables_initializer())
                        tfp = outputs.eval()

                        assert Xp.tolist() == tfp.tolist(), (tfp - Xp)

def test_simple_construct():
    input_shape = [None, 28, 28, 1]

    for net_name in ['simplewide', 'simple']:
        layers = construct(net_name, input_shape)
        metadata = constructed_metadata(net_name, input_shape)

        for i, layer in enumerate(layers):
            if layer['type'] == 'convolution_2d':
                if net_name != 'simplewide':
                    assert np.array(layer['kernel']).shape[-1] == 64

                assert layers[i + 1]['type'] == 'batch_normalization'
                assert layers[i + 2]['type'] == 'activation'

        assert metadata['base_image_network'] == net_name
        assert metadata['input_image_shape'] == input_shape[1:]

        if net_name == 'simplewide':
            metadata['outputs'] == 512
        else:
            metadata['outputs'] == 64
