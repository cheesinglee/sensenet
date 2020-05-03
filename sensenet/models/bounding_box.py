import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()
kl = sensenet.importers.import_keras_layers()

from sensenet.constants import MAX_BOUNDING_BOXES, MASKS
from sensenet.constants import IGNORE_THRESHOLD, IOU_THRESHOLD
from sensenet.accessors import number_of_classes, get_anchors, get_image_shape
from sensenet.layers.construct import layer_sequence
from sensenet.layers.utils import constant, shape
from sensenet.models.settings import ensure_settings
from sensenet.preprocess.image import ImageReader, ImageLoader
from sensenet.pretrained import load_pretrained_weights

def branch_head(feats, anchors, num_classes, input_shape, calc_loss):
    nas = len(anchors)

    # Reshape to batch, height, width, nanchors, box_params.
    at = tf.reshape(constant(anchors, tf.float32), [1, 1, 1, nas, 2])

    grid_shape = np.array(shape(feats)[1:3], dtype=np.int32) # height, width
    x_shape = grid_shape[1]
    y_shape = grid_shape[0]
    x_range = tf.range(0, x_shape)
    y_range = tf.range(0, y_shape)

    grid_x = tf.tile(tf.reshape(x_range, [1, -1, 1, 1]), [y_shape, 1, 1, 1])
    grid_y = tf.tile(tf.reshape(y_range, [-1, 1, 1, 1]), [1, x_shape, 1, 1])
    grid = tf.cast(tf.concat([grid_x, grid_y], -1), feats.dtype)

    feats = tf.reshape(feats, [-1, y_shape, x_shape, nas, num_classes + 5])

    t_grid = constant(grid_shape[::-1], feats.dtype)
    t_input = constant(input_shape[::-1], feats.dtype)

    # Adjust predictions to each spatial grid point and anchor size.
    box_xy = (tf.sigmoid(feats[..., :2]) + grid) / t_grid
    box_wh = tf.exp(feats[..., 2:4]) * at / t_input
    box_confidence = tf.sigmoid(feats[..., 4:5])
    box_class_probs = tf.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    else:
        return box_xy, box_wh, box_confidence, box_class_probs

class BoxLocator(tf.keras.layers.Layer):
    def __init__(self, network, nclasses, settings):
        super(BoxLocator, self).__init__()

        self._nclasses = nclasses
        self._threshold = settings.bounding_box_threshold or IGNORE_THRESHOLD
        self._iou_threshold = settings.iou_threshold or IOU_THRESHOLD
        self._anchors = get_anchors(network)

    def correct_boxes(self, box_xy, box_wh, input_shape):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        input_shape = constant(input_shape, box_yx.dtype)
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)

        min_maxes = [box_mins[..., 0:1],   # y_min
                     box_mins[..., 1:2],   # x_min
                     box_maxes[..., 0:1],  # y_max
                     box_maxes[..., 1:2]]  # x_max

        boxes =  tf.concat(min_maxes, -1)
        boxes *= tf.concat([input_shape, input_shape], -1)

        return boxes

    def branch_head(self, features, anchors, input_shape):
        return branch_head(features, anchors, self._nclasses, input_shape, False)

    def boxes_and_scores(self, features, anchors, input_shape):
        xy, wh, conf, probs = self.branch_head(features, anchors, input_shape)
        boxes = tf.reshape(self.correct_boxes(xy, wh, input_shape), [-1, 4])
        box_scores = tf.reshape(conf * probs, [-1, self._nclasses])

        return boxes, box_scores

    def call(self, inputs):
        input_shape = shape(inputs[0])[1:3] * 32

        boxes = []
        box_scores = []

        for features, anchors in zip(inputs, self._anchors):
            bxs, scs = self.boxes_and_scores(features, anchors, input_shape)
            boxes.append(bxs)
            box_scores.append(scs)

        boxes = tf.concat(boxes, 0)
        box_scores = tf.concat(box_scores, 0)

        mask = box_scores >= self._threshold
        max_boxes = constant(MAX_BOUNDING_BOXES, tf.int32)

        boxes_ = []
        scores_ = []
        classes_ = []

        for c in range(self._nclasses):
            c_boxes = tf.boolean_mask(boxes, mask[:, c])
            c_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

            iou_t = self._iou_threshold
            nms = tf.image.non_max_suppression
            nms_index = nms(c_boxes, c_scores, max_boxes, iou_threshold=iou_t)

            c_boxes = tf.gather(c_boxes, nms_index)
            c_scores = tf.gather(c_scores, nms_index)
            classes = tf.ones_like(c_scores, tf.int32) * c

            boxes_.append(c_boxes)
            scores_.append(c_scores)
            classes_.append(classes)

        return (tf.expand_dims(tf.concat(boxes_, 0), 0, name='boxes'),
                tf.expand_dims(tf.concat(scores_, 0), 0, name='scores'),
                tf.expand_dims(tf.concat(classes_, 0), 0, name='classes'))

def box_detector(model, input_settings):
    settings = ensure_settings(input_settings)

    network = model['image_network']
    reader = ImageReader(network, settings)

    if settings.input_image_format == 'pixel_values':
        image_shape = get_image_shape(model)
        image_input = kl.Input(image_shape[1:], dtype=tf.float32, name='image')
        raw_image = reader(image_input)
    else:
        image_input = kl.Input((1,), dtype=tf.string, name='image')
        raw_image = reader(image_input[:,0])

    loader = ImageLoader(network)
    yolo_tail = layer_sequence(network)
    locator = BoxLocator(network, number_of_classes(model), settings)

    image = loader(raw_image)
    features = yolo_tail(image)

    # Boxes, scores, and classes
    all_outputs = locator(features)
    return tf.keras.Model(inputs=image_input, outputs=all_outputs)
