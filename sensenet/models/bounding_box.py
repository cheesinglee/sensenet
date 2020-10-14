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

        assert network['layers'][-1]['type'] == 'yolo_output_branches'

        self._nclasses = nclasses
        self._input_shape = get_image_shape(network)[1:3]

        ob = network['layers'][-1]['output_branches']
        self._strides = tuple([self._input_shape[0] // b['strides'] for b in ob])
        self._nanchors = tuple([len(b['anchors']) for b in ob])

        self._unfiltered = settings.output_unfiltered_boxes
        self._threshold = settings.bounding_box_threshold or SCORE_THRESHOLD
        self._iou_threshold = settings.iou_threshold or IOU_THRESHOLD
        self._max_objects = settings.max_objects or MAX_OBJECTS

    def boxes_min_max(self, box_xy, box_wh):
        input_shape = tf.constant(self._input_shape, dtype=tf.float32)

        box_yx = box_xy[:,:,::-1]
        box_hw = box_wh[:,:,::-1]

        box_mins = (box_yx - (box_hw / 2.)) / input_shape
        box_maxes = (box_yx + (box_hw / 2.)) / input_shape

        return  tf.concat([
            box_mins[:,:,0:1],  # y_min
            box_mins[:,:,1:2],  # x_min
            box_maxes[:,:,0:1],  # y_max
            box_maxes[:,:,1:2]  # x_max
        ], axis=-1)

    def reshape3d(self, x, nrows, ncols):
        return tf.reshape(x, (tf.shape(x)[0], nrows, ncols))

    def call(self, predictions, original_shape):
        box_bounds = []
        box_scores = []

        for pred, stride, na in zip(predictions, self._strides, self._nanchors):
            bboxes = pred[1]
            nboxes = stride * stride * na

            map_boxes = bboxes[:,:,:,:,:4]
            map_scores = bboxes[:,:,:,:,4:5] * bboxes[:,:,:,:,5:]

            box_bounds.append(self.reshape3d(map_boxes, nboxes, 4))
            box_scores.append(self.reshape3d(map_scores, nboxes, self._nclasses))

        scores = tf.concat(box_scores, axis=1, name='all_scores')
        all_boxes = tf.concat(box_bounds, axis=1, name='all_boxes')

        box_xy, box_wh = tf.split(all_boxes, (2, 2), axis=-1)
        boxes = self.boxes_min_max(box_xy, box_wh)

        # Here we're assuming that we only get one image at a time as input
        # I'm not sure this can be vectorized
        nboxes = tf.shape(scores)[1]
        img_boxes = tf.reshape(boxes, (nboxes, 4))
        img_scores = tf.reshape(scores, (nboxes, self._nclasses))

        max_dim = tf.reduce_max(original_shape[0][:2], axis=-1, name='mdim')
        scaled_boxes = tf.math.round(img_boxes * max_dim, name='boxes')
        classes = tf.cast(tf.argmax(img_scores, axis=1), tf.int32)

        scores_shape = tf.shape(img_scores)
        score_idxs = classes + (tf.range(nboxes) * self._nclasses)
        max_box_scores = tf.gather(tf.reshape(img_scores, (-1,)), score_idxs)

        if self._unfiltered:
            selected_boxes = scaled_boxes
            selected_scores = max_box_scores
            selected_classes = classes
        else:
            selected_indices, num_valid = tf.image.non_max_suppression_padded(
                boxes=scaled_boxes,
                scores=max_box_scores,
                max_output_size=self._max_objects,
                iou_threshold=self._iou_threshold,
                score_threshold=self._threshold)

            selected_boxes = tf.gather(scaled_boxes, selected_indices)
            selected_scores = tf.gather(max_box_scores, selected_indices)
            selected_classes = tf.gather(classes, selected_indices)

        xyxy_boxes = tf.gather(selected_boxes, [1,0,3,2], axis=-1)

        final_boxes = tf.expand_dims(xyxy_boxes, axis=0)
        final_scores = tf.expand_dims(selected_scores, axis=0)
        final_classes = tf.expand_dims(selected_classes, axis=0)

        return final_boxes, final_scores, final_classes

def box_detector(model, input_settings):
    settings = ensure_settings(input_settings)

    network = model['image_network']
    reader = BoundingBoxImageReader(network, settings)
    nclasses = number_of_classes(model)
    loader = ImageLoader(network)

    yolo_trunk = YoloTrunk(network, nclasses)
    yolo_branches = YoloBranches(network, nclasses)
    locator = BoxLocator(network, nclasses, settings)

    if settings.input_image_format == 'pixel_values':
        image_input = kl.Input((None, None, 3), dtype=tf.float32, name='image')
    else:
        image_input = kl.Input((1,), dtype=tf.string, name='image')

    raw_image, original_shape = reader(image_input)
    image = loader(raw_image)
    layer_outputs = yolo_trunk(image)
    predictions = yolo_branches(layer_outputs)
    all_outputs = locator(predictions, tf.cast(original_shape, tf.float32))

    return tf.keras.Model(inputs=image_input, outputs=all_outputs)
