import sensenet.importers

from keras.models import InputLayer, Model
from keras.layers import Input, MaxPooling2D, AveragePooling2D, Dense
from keras.layers import BatchNormalization, Conv2D, Activation
from keras.layers import GlobalAveragePooling2D

from sensenet.graph.keras.extract import extract

def dims(layer):
    out_shape = []

    for d in layer.get_shape():
        try:
            out_shape.append(int(d))
        except TypeError:
            out_shape.append(None)

    return out_shape

def conv_bn_act(input_layer, nfilters, shape):
    x = Conv2D(nfilters, shape, padding='same')(input_layer)
    x = BatchNormalization()(x)
    return Activation('relu')(x)

def finish(top, nfilters):
    current_hw = tuple(dims(top)[1:3])

    x = Conv2D(nfilters, current_hw)(top)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Make sure we're left with a single output "pixel"
    assert dims(x)[:-1] == [None, 1, 1]

    return GlobalAveragePooling2D()(x)

def create_simple(shape):
    if max(shape[0], shape[1]) >= 64:
        filter_shape = (5, 5)
        pool_op = (4, 4)
    else:
        filter_shape = (3, 3)
        pool_op = (2, 2)

    x = inputs = Input(shape=shape)

    for _ in range(3):
        x = conv_bn_act(x, 64, filter_shape)
        x = MaxPooling2D(pool_size=pool_op)(x)

    return inputs, finish(x, 64)

def create_simple_wide(shape):
    if max(shape[0], shape[1]) >= 64:
        current_filters = 32
    else:
        current_filters = 64

    x = inputs = Input(shape=shape)
    current_shape = shape

    while current_shape[0] > 3 or current_shape[1] > 3:
        x = conv_bn_act(x, current_filters, (3, 3))
        x = MaxPooling2D(pool_size=(2, 2))(x)

        current_shape = dims(x)[1:]
        current_filters = min(2 * current_filters, 512)

    return inputs, finish(x, current_filters)

def build_layers(base_network, input_shape):
    if base_network == 'simple':
        net_method = create_simple
    elif base_network == 'simplewide':
        net_method = create_simple_wide
    else:
        raise ValueError('Network type "%s" unknown' % base_network)

    return net_method(input_shape[1:])

def construct(base_network, input_shape):
    inputs, outputs = build_layers(base_network, input_shape)
    net = Model(inputs, outputs)

    assert type(net.layers[0]) == InputLayer

    return [extract(layer) for layer in net.layers[1:]]

def constructed_metadata(base_network, input_shape):
    image_shape = [input_shape[2], input_shape[1], input_shape[3]]
    _, outputs = build_layers(base_network, input_shape)
    out_dims = dims(outputs)

    assert len(out_dims) == 2 and out_dims[0] is None

    outputs = out_dims[1]

    return {
        'base_image_network': base_network,
        'input_image_shape': image_shape,
        'version': None,
        'loading_method': 'normalizing',
        'outputs': outputs
    }
