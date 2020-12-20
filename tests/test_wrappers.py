import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

import os
import json
import gzip

from PIL import Image

from sensenet.models.wrappers import create_model

from .utils import TEST_DATA_DIR, TEST_IMAGE_DATA
from .test_pretrained import check_image_prediction

BUS_INDEX = 779
BUS_PATH = os.path.join(TEST_IMAGE_DATA, 'bus.jpg')

MOBILENET_PATH = os.path.join(TEST_DATA_DIR, 'mobilenetv2.json.gz')

def make_mobilenet(settings):
    with gzip.open(MOBILENET_PATH, 'rb') as fin:
        network = json.load(fin)

    return create_model(network, settings)

def check_pixels_and_file(settings, pos_threshold, neg_threshold):
    pix_settings = dict(settings)
    pix_settings['input_image_format'] = 'pixel_values'
    pix_model = make_mobilenet(pix_settings)

    file_settings = dict(settings)
    file_settings['input_image_format'] = 'file'
    file_model = make_mobilenet(file_settings)

    image_pixels = np.array(Image.open(BUS_PATH))

    if pix_settings.get('color_space', '').endswith('a'):
        new_pixels = np.zeros((image_pixels.shape[0], image_pixels.shape[1], 4))
        new_pixels[:,:,:3] = image_pixels
        image_pixels = new_pixels

    pixel_pred = pix_model(image_pixels)
    file_pred = file_model(BUS_PATH)

    diffs = np.abs(pixel_pred - file_pred)
    assert np.all(diffs < 1e-3), np.max(diffs)

    for pred in [pixel_pred, file_pred]:
        check_image_prediction(pred, BUS_INDEX, pos_threshold, neg_threshold)

    return pixel_pred

def test_channel_order():
    for space, pos, neg in [('rgb', 0.98, 0.02), ('bgr', 0.11, 0.5)]:
        settings = {'color_space': space}
        no_alpha = check_pixels_and_file(settings, pos, neg)

        settings = {'color_space': space + 'a'}
        alpha = check_pixels_and_file(settings, pos, neg)

        assert np.all(np.abs(alpha - no_alpha) < 1e-4)

def test_cropping():
    for rt, threshold in [('warp', 0.98), ('pad', 0.96), ('crop', 0.97)]:
        settings = {'rescale_type': rt}
        pred = check_pixels_and_file(settings, threshold, 0.02)

        assert threshold < pred[0, BUS_INDEX] < threshold + 0.01
