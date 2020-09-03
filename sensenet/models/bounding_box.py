import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()
kl = sensenet.importers.import_keras_layers()

from sensenet.constants import MAX_OBJECTS
from sensenet.constants import SCORE_THRESHOLD, IGNORE_THRESHOLD, IOU_THRESHOLD
from sensenet.accessors import number_of_classes, get_image_shape
from sensenet.layers.yolo import YoloTrunk, YoloBranches
from sensenet.models.settings import ensure_settings
from sensenet.preprocess.image import BoundingBoxImageReader, ImageLoader
from sensenet.pretrained import load_pretrained_weights

class BoxLocator(tf.keras.layers.Layer):
    def __init__(self, network, nclasses, settings):
        super(BoxLocator, self).__init__()

        self._nclasses = nclasses
        self._input_shape = get_image_shape(network)[1:3]

        self._threshold = settings.bounding_box_threshold or SCORE_THRESHOLD
        self._iou_threshold = settings.iou_threshold or IOU_THRESHOLD
        self._max_objects = settings.max_objects or MAX_OBJECTS

    def filter_boxes(self, box_xywh, scores, limits):
        scores_max = tf.math.reduce_max(scores, axis=-1)
        mask = scores_max >= min(0.2, self._threshold)

        boxes_masked = tf.boolean_mask(box_xywh, mask)
        boxes_shape = (tf.shape(scores)[0], -1, tf.shape(boxes_masked)[-1])
        class_boxes = tf.reshape(boxes_masked, boxes_shape)

        preds_masked = tf.boolean_mask(scores, mask)
        conf_shape = (tf.shape(scores)[0], -1, tf.shape(preds_masked)[-1])
        pred_conf = tf.reshape(preds_masked, conf_shape)

        box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)
        input_shape = tf.constant(self._input_shape, dtype=tf.float32)

        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        box_mins = (box_yx - (box_hw / 2.)) / input_shape
        box_maxes = (box_yx + (box_hw / 2.)) / input_shape
        boxes = tf.concat([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ], axis=-1)

        ylim, xlim = limits[0], limits[1]

        amask = tf.logical_and(box_xy[..., 0] < xlim, box_xy[..., 1] < ylim)
        boxes = tf.reshape(tf.boolean_mask(boxes, amask), boxes_shape)
        pred_conf = tf.reshape(tf.boolean_mask(pred_conf, amask), conf_shape)

        return boxes, pred_conf

    def reshape3d(self, x):
        return tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-1]))

    def call(self, predictions, original_shape):
        box_bounds = []
        box_scores = []

        for _, bboxes in predictions:
            map_boxes = bboxes[..., :4]
            map_scores = bboxes[..., 4:5] * bboxes[..., 5:]

            box_bounds.append(self.reshape3d(map_boxes))
            box_scores.append(self.reshape3d(map_scores))

        boxes = tf.concat(box_bounds, axis=1)
        scores = tf.concat(box_scores, axis=1)

        # Here we're assuming that we only get one image at a time as input
        # I'm not sure this can be vectorized, given the flattening that
        # happens when we mask above
        max_dim = tf.reduce_max(original_shape[0][:2])
        limits = self._input_shape * original_shape[0][:2] / max_dim

        boxes, scores = self.filter_boxes(boxes, scores, limits)
        sc_shape = tf.shape(scores)

        boxes, scores, classes, valid = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(scores, (sc_shape[0], -1, sc_shape[-1])),
            max_output_size_per_class=self._max_objects,
            max_total_size=self._max_objects,
            iou_threshold=self._iou_threshold,
            score_threshold=self._threshold
        )

        vboxes = tf.gather(boxes[0,:valid[0],...], [1,0,3,2], axis=-1)
        vboxes = tf.math.round(vboxes * max_dim)

        return vboxes, scores[0,:valid[0]], classes[0,:valid[0]]

def box_detector(model, input_settings):
    settings = ensure_settings(input_settings)

    network = model['image_network']
    reader = BoundingBoxImageReader(network, settings)

    if settings.input_image_format == 'pixel_values':
        image_shape = get_image_shape(model)
        image_input = kl.Input(image_shape[1:], dtype=tf.float32, name='image')
        raw_image, original_shape = reader(image_input)
    else:
        image_input = kl.Input((1,), dtype=tf.string, name='image')
        raw_image, original_shape = reader(image_input[:,0])

    nclasses = number_of_classes(model)
    loader = ImageLoader(network)

    yolo_trunk = YoloTrunk(network, nclasses)
    yolo_branches = YoloBranches(network, nclasses)
    locator = BoxLocator(network, nclasses, settings)

    image = loader(raw_image)
    layer_outputs = yolo_trunk(image)
    predictions = yolo_branches(layer_outputs)
    all_outputs = locator(predictions, tf.cast(original_shape, tf.float32))

    return tf.keras.Model(inputs=image_input, outputs=all_outputs)
