import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()
kl = sensenet.importers.import_keras_layers()

from sensenet.constants import MAX_BOUNDING_BOXES, MASKS
from sensenet.constants import IGNORE_THRESHOLD, IOU_THRESHOLD
from sensenet.accessors import number_of_classes, get_anchors
from sensenet.pretrained import complete_image_network
from sensenet.layers.construct import LAYER_FUNCTIONS
from sensenet.layers.utils import constant, propagate, make_sequence
from sensenet.preprocess.image import ImageReader, ImageLoader

def shape(tensor):
    return np.array(tensor.get_shape().as_list(), dtype=np.float32)

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


class YoloTail(tf.keras.layers.Layer):
    def __init__(self, network):
        super(YoloTail, self).__init__()

        self._trunk = []
        self._branches = []
        self._concatenations = {}

        for i, layer in enumerate(network['layers'][:-1]):
            ltype = layer['type']
            self._trunk.append(LAYER_FUNCTIONS[ltype](layer))

            if ltype == 'concatenate':
                self._concatenations[i] = layer['inputs']

        assert network['layers'][-1]['type'] == 'yolo_output_branches'
        out_branches = network['layers'][-1]

        for i, branch in enumerate(out_branches['output_branches']):
            idx = branch['input']
            layers = make_sequence(branch['convolution_path'], LAYER_FUNCTIONS)

            self._branches.append((idx, layers))

    def call(self, inputs):
        outputs = []
        next_inputs = inputs

        for i, layer in enumerate(self._trunk):
            if i in self._concatenations:
                inputs = self._concatenations[i]
                next_inputs = layer([outputs[j] for j in inputs])
            else:
                next_inputs = layer(next_inputs)

            outputs.append(next_inputs)

        return [propagate(layers, outputs[i]) for i, layers in self._branches]


class BoxLocator(tf.keras.layers.Layer):
    def __init__(self, network, nclasses, extras):
        super(BoxLocator, self).__init__()

        self._nclasses = nclasses
        self._threshold = extras.get('bounding_box_threshold', IGNORE_THRESHOLD)
        self._iou_threshold = extras.get('iou_threshold', IOU_THRESHOLD)
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

def box_detector(model, extras):
    network = complete_image_network(model['image_network'])
    image_input = kl.Input((1,), dtype=tf.string, name='image')

    reader = ImageReader(network, extras)
    loader = ImageLoader(network)
    yolo_tail = YoloTail(network)
    locator = BoxLocator(network, number_of_classes(model), extras)

    raw_image = reader(image_input[:,0])
    image = loader(raw_image)
    features = yolo_tail(image)

    boxes, scores, classes = locator(features)

    return tf.keras.Model(inputs=image_input, outputs=[boxes, scores, classes])