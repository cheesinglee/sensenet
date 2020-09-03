# Data and objective types
NUMERIC = 'numeric'
CATEGORICAL = 'categorical'
IMAGE_PATH = 'image'
TRIPLET = 'triplet'
BOUNDING_BOX = 'bounding_box'

# Keys for numeric preprocessors
MEAN = 'mean'
STANDARD_DEVIATION = 'stdev'
ZERO = 'zero_value'
ONE = 'one_value'

# Constants for the scaled linear units
SELU_ALPHA = 1.6732632423543772848170429916717;
SELU_SCALE = 1.0507009873554804934193349852946;

# For the leaky ReLU, we need to match the keras default
LEAKY_RELU_ALPHA = 0.1

# Pixel normalizing factors for pretrained networks
CAFFE_MEAN = [103.939, 116.779, 123.68]
TORCH_MEAN = [c * 255 for c in [0.485, 0.456, 0.406]]
TORCH_STD = [c * 255 for c in [0.229, 0.224, 0.225]]

IMAGE_STANDARDIZERS = {
    None: (0., 1.),
    'normalizing': (0, 255.),
    'centering': (127.5, 127.5),
    'skewed_centering': (128., 128.),
    'channelwise_centering': (CAFFE_MEAN, 1.),
    'channelwise_standardizing': (TORCH_MEAN, TORCH_STD)
}

# Default parameters for YOLO bounding box detection
MAX_OBJECTS = 32
SCORE_THRESHOLD = 0.5
IGNORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
