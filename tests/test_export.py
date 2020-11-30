import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

import os
import shutil

from PIL import Image

from sensenet.models.wrappers import to_tflite, to_tfjs

from .utils import TEST_DATA_DIR, TEST_IMAGE_DATA
from .test_pretrained import create_image_model

TEST_SAVE_MODEL = os.path.join(TEST_DATA_DIR, 'test_model_save')
TEST_TF_LITE_MODEL = os.path.join(TEST_SAVE_MODEL, 'model.tflite')

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

def test_tf_js_classifier():
    shutil.rmtree(TEST_SAVE_MODEL, ignore_errors=True)

    pixel_input = {'input_image_format': 'pixel_values'}
    model = create_image_model('mobilenetv2', pixel_input)

    to_tfjs(model, TEST_SAVE_MODEL)

    # Commenting this delete out will allow you to test the exported
    # model with nodejs/tensorflowjs/canvas, if you have these things
    # installed, using the test_model.js script in this directory, by
    # navigating to sensenet/tests and running `node test_model.js`.
    shutil.rmtree(TEST_SAVE_MODEL, ignore_errors=True)

def test_tf_js_boxes():
    shutil.rmtree(TEST_SAVE_MODEL, ignore_errors=True)

    pixel_input = {'input_image_format': 'pixel_values'}
    model = create_image_model('tinyyolov4', pixel_input)

    to_tfjs(model, TEST_SAVE_MODEL)

    # As above, you can comment this out to test in JS, but here you
    # must also set the `bounding_boxes` variable to true in the test
    # script.
    shutil.rmtree(TEST_SAVE_MODEL, ignore_errors=True)
