import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.models.bounding_box import box_detector
from sensenet.models.deepnet import deepnet_model
from sensenet.preprocess.preprocessor import Preprocessor
from sensenet.pretrained import get_image_network, load_pretrained_weights

def image_model(network, settings):
    if 'yolo' in network['image_network']['metadata']['base_image_network']:
        model = box_detector(network, settings)
    else:
        model = deepnet_model(network, settings)

    if settings.load_pretrained_weights:
        load_pretrained_weights(model, network['image_network'])

    return model

def pretrained_image_model(network_name, settings):
    network = get_image_network(network_name)
    return image_model(network, settings)

def get_layer(model, layer_type, names):
    for layer in model.layers:
        if type(layer) == layer_type:
            if names is None or layer.name in names:
                return layer

    msg = 'Could not find layer of type %s' % str(layer_type)

    if names is not None:
        msg += 'with name in %s' % str(names)

    raise ValueError(msg)

def io_for_extractor(model):
    preprocessor = get_layer(model, Preprocessor, None)
    return preprocessor.input, preprocessor.output

def image_feature_extractor(model):
    image_input, features = io_for_extractor(model)
    return tf.keras.Model(inputs=image_input, outputs=features)
