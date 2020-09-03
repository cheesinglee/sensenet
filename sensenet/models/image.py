import sensenet.importers
tf = sensenet.importers.import_tensorflow()

import math

from sensenet.accessors import get_layer, is_yolo_model
from sensenet.layers.yolo import YoloTrunk
from sensenet.models.bounding_box import box_detector
from sensenet.models.deepnet import deepnet_model
from sensenet.models.settings import ensure_settings
from sensenet.preprocess.preprocessor import Preprocessor
from sensenet.pretrained import load_pretrained_weights, get_pretrained_network

def image_model(network, input_settings):
    settings = ensure_settings(input_settings)

    if is_yolo_model(network):
        model = box_detector(network, settings)
    else:
        model = deepnet_model(network, settings)

    if settings.load_pretrained_weights:
        load_pretrained_weights(model, network['image_network'])

    return model

def pretrained_image_model(network_name, settings):
    network = get_pretrained_network(network_name)
    return image_model(network, settings)

def io_for_extractor(model):
    preprocessor = get_layer(model, Preprocessor, None)
    return preprocessor.input, preprocessor.output

def image_feature_extractor(model):
    image_input, features = io_for_extractor(model)
    return tf.keras.Model(inputs=image_input, outputs=features)

def image_layers(model):
    try:
        yolo_trunk = get_layer(model, YoloTrunk, None)
        layers = yolo_trunk._trunk
    except ValueError:
        preprocessor = get_layer(model, Preprocessor, None)
        layers = preprocessor.get_image_layers()

    return layers
