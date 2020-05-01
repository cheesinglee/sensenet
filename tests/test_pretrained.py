import json
import gzip

import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import CATEGORICAL, IMAGE_PATH, BOUNDING_BOX

from sensenet.load import load_points
from sensenet.models.image import pretrained_image_model, image_feature_extractor
from sensenet.preprocess.image import ImageReader
from sensenet.pretrained import get_image_network
from sensenet.models.settings import Settings

EXTRA_PARAMS = {
    'image_path_prefix': 'tests/data/images/',
    'load_pretrained_weights': True
}

def create_image_model(network_name, box_threshold, image_format):
    extras = Settings(EXTRA_PARAMS)
    extras.input_image_format = image_format
    extras.bounding_box_threshold = box_threshold

    return pretrained_image_model(network_name, extras)

def reader_for_network(network_name):
    network = get_image_network(network_name)
    read_settings = Settings(EXTRA_PARAMS)

    return ImageReader(network['image_network'], read_settings).get_reader_fn()

def classify(network_name, accuracy_threshold):
    network = get_image_network(network_name)
    noutputs = network['image_network']['metadata']['outputs']

    image_model = create_image_model(network_name, None, 'file')
    pixel_model = create_image_model(network_name, None, 'pixel_values')
    read = reader_for_network(network_name)

    image_feature_extractor(pixel_model)
    ex_mod = image_feature_extractor(image_model)

    for image, cidx in [('dog.jpg', 254), ('bus.jpg', 779)]:
        point = load_points(network, [[image]])
        file_pred = image_model.predict(point)

        img_px = np.expand_dims(read(image).numpy(), axis=0)
        pixel_pred = pixel_model.predict(img_px)

        for pred in [file_pred, pixel_pred]:
            for i, p in enumerate(pred.flatten().tolist()):
                if i == cidx:
                    assert p > accuracy_threshold, str((i, p))
                else:
                    assert p < 0.02, str((i, p))

    outputs = ex_mod(load_points(network, [['dog.jpg'], ['bus.jpg']]))
    assert outputs.shape == (2, noutputs)

def test_resnet50():
    classify('resnet50', 0.99)

def test_mobilenet():
    classify('mobilenet', 0.97)

def test_xception():
    classify('xception', 0.88)

def test_resnet18():
    classify('resnet18', 0.96)

def test_mobilenetv2():
    classify('mobilenetv2', 0.88)

def detect_bounding_boxes(network_name, nboxes, class_list, threshold):
    image_detector = create_image_model(network_name, threshold, 'file')
    pixel_detector = create_image_model(network_name, threshold, 'pixel_values')
    read = reader_for_network(network_name)

    file_pred = image_detector.predict([['pizza_people.jpg']])
    img_px = np.expand_dims(read('pizza_people.jpg').numpy(), axis=0)
    pixel_pred = pixel_detector.predict(img_px)

    for pred in [file_pred, pixel_pred]:
        boxes, scores, classes = pred
        assert len(boxes[0]) == nboxes
        assert sorted(set(classes[0])) == sorted(class_list), str(set(classes[0]))

def test_yolov3():
    detect_bounding_boxes('yolov3', 6, [0, 60, 53], 0.6)

def test_tinyyolov3():
    detect_bounding_boxes('tinyyolov3', 3, [0, 53], 0.4)
