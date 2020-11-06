import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

import json
import gzip
import os
import shutil

from PIL import Image, ImageDraw

from sensenet.constants import CATEGORICAL, IMAGE_PATH, BOUNDING_BOX

from sensenet.accessors import get_image_shape
from sensenet.load import load_points
from sensenet.models.image import pretrained_image_model, image_feature_extractor
from sensenet.models.image import image_layers, get_pretrained_network
from sensenet.models.settings import Settings
from sensenet.models.wrappers import to_tflite
from sensenet.preprocess.image import get_image_reader_fn

from .utils import TEST_DATA_DIR, TEST_IMAGE_DATA

TEST_SAVE_MODEL = os.path.join(TEST_DATA_DIR, 'test_model_save')
TEST_TF_LITE_MODEL = os.path.join(TEST_SAVE_MODEL, 'model.tflite')

EXTRA_PARAMS = {
    'bounding_box_threshold': 0.5,
    'image_path_prefix': TEST_IMAGE_DATA,
    'input_image_format': 'file',
    'load_pretrained_weights': True
}

def create_image_model(network_name, additional_settings):
    extras = dict(EXTRA_PARAMS)

    if additional_settings:
        extras.update(additional_settings)

    return pretrained_image_model(network_name, Settings(extras))

def reader_for_network(network_name):
    image_shape = get_image_shape(get_pretrained_network(network_name))
    path_prefix = EXTRA_PARAMS['image_path_prefix']

    return get_image_reader_fn(image_shape, 'file', path_prefix)

def classify(network_name, accuracy_threshold):
    pixel_input = {'input_image_format': 'pixel_values'}

    network = get_pretrained_network(network_name)
    nlayers = len(network['image_network']['layers'])
    noutputs = network['image_network']['metadata']['outputs']
    preprocessors = network['preprocess']

    image_model = create_image_model(network_name, None)
    pixel_model = create_image_model(network_name, pixel_input)
    read = reader_for_network(network_name)

    assert len(image_layers(pixel_model)) == nlayers

    # Just check if this is possible
    image_feature_extractor(pixel_model)
    ex_mod = image_feature_extractor(image_model)

    for image, cidx in [('dog.jpg', 254), ('bus.jpg', 779)]:
        point = load_points(preprocessors, [[image]])
        file_pred = image_model.predict(point)

        img_px = np.expand_dims(read(image).numpy(), axis=0)
        pixel_pred = pixel_model.predict(img_px)

        for pred in [file_pred, pixel_pred]:
            for i, p in enumerate(pred.flatten().tolist()):
                if i == cidx:
                    assert p > accuracy_threshold, str((i, p))
                else:
                    assert p < 0.02, str((i, p))

    outputs = ex_mod(load_points(preprocessors, [['dog.jpg'], ['bus.jpg']]))
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
    file_input = {'bounding_box_threshold': threshold}

    pixel_input = {
        'input_image_format': 'pixel_values',
        'bounding_box_threshold': threshold
    }

    image_detector = create_image_model(network_name, file_input)
    pixel_detector = create_image_model(network_name, pixel_input)
    read = reader_for_network(network_name)

    file_pred = image_detector.predict([['pizza_people.jpg']])
    img_px = np.expand_dims(read('pizza_people.jpg').numpy(), axis=0)
    pixel_pred = pixel_detector.predict(img_px)

    for pred in [file_pred, pixel_pred]:
        boxes, scores, classes = pred[0][0], pred[1][0], pred[2][0]

        assert len(boxes) == len(scores) == nboxes, len(boxes)
        assert sorted(set(classes)) == sorted(class_list), classes

def test_tinyyolov4():
    detect_bounding_boxes('tinyyolov4', 5, [0, 53], 0.4)

def tflite_predict(model, len_inputs, len_outputs, test_file):
    shutil.rmtree(TEST_SAVE_MODEL, ignore_errors=True)
    os.makedirs(TEST_SAVE_MODEL)

    to_tflite(model, TEST_TF_LITE_MODEL)
    img = Image.open(os.path.join(TEST_IMAGE_DATA, test_file))
    in_shape = [1] + list(img.size)[::-1] + [3]

    interpreter = tf.lite.Interpreter(model_path=TEST_TF_LITE_MODEL)
    interpreter.resize_tensor_input(0, in_shape, strict=True)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    assert len(input_details) == len_inputs
    assert len(output_details) == len_outputs

    input_data = np.expand_dims(img.convert('RGB'), axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    return [interpreter.get_tensor(od['index']) for od in output_details]

def test_tf_lite_deepnet():
    pixel_input = {'input_image_format': 'pixel_values'}
    model = create_image_model('mobilenetv2', pixel_input)
    probs = tflite_predict(model, 1, 1, 'dog.jpg')

    assert len(probs) == 1
    assert probs[0].shape == (1, 1000), probs[0].shape
    # It's generous but the math here is 32-bit
    assert abs(np.sum(probs[0]) - 1) < 1e-5, np.sum(probs[0])
    assert probs[0][0, 254] > 0.88

    shutil.rmtree(TEST_SAVE_MODEL)

def test_tf_lite_boxes():
    pixel_input = {
        'input_image_format': 'pixel_values',
        'output_unfiltered_boxes': True
    }

    detector = create_image_model('tinyyolov4', pixel_input)
    boxes, scores, classes = tflite_predict(detector, 1, 3, 'strange_car.png')

    assert boxes.shape == (1, 2535, 4), boxes.shape
    assert classes.shape == (1, 2535), classes.shape
    assert scores.shape == (1, 2535), scores.shape

    for box, cls, score in zip(boxes[0], classes[0], scores[0]):
        # There are a few boxes here, but they should all find the
        # same thing in roughly the same place
        if score > 0.5:
            assert 550 < box[0] < 600, box
            assert 220 < box[1] < 270, box
            assert 970 < box[2] < 1020, box
            assert 390 < box[3] < 440, box

            assert cls == 2

    shutil.rmtree(TEST_SAVE_MODEL)

def test_empty():
    detector = create_image_model('tinyyolov4', None)
    boxes, scores, classes  = detector.predict([['black.png']])

    assert len(boxes[0]) == 0
    assert len(scores[0]) == 0
    assert len(classes[0]) == 0

def test_scaling():
    detector = create_image_model('tinyyolov4', None)
    boxes, scores, classes  = detector.predict([['strange_car.png']])

    assert 550 < boxes[0, 0, 0] < 600,  boxes[0, 0]
    assert 220 < boxes[0, 0, 1] < 270,  boxes[0, 0]
    assert 970 < boxes[0, 0, 2] < 1020,  boxes[0, 0]
    assert 390 < boxes[0, 0, 3] < 440,  boxes[0, 0]

    assert scores[0] > 0.9
    assert classes[0] == 2
