import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import MAX_BOUNDING_BOXES, MASKS
from sensenet.graph.image import complete_image_network, graph_input_shape
from sensenet.graph.image import normalize_image
from sensenet.graph.construct import make_all_outputs, yolo_output_branches
from sensenet.graph.layers.utils import make_tensor

def shape(tensor):
    return np.array(tensor.get_shape().as_list(), dtype=np.float32)

def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    nas = len(anchors)

    # Reshape to batch, height, width, nanchors, box_params.
    at = tf.reshape(make_tensor(anchors, ttype=feats.dtype), [1, 1, 1, nas, 2])

    grid_shape = np.array(shape(feats)[1:3], dtype=np.int32) # height, width
    x_shape = grid_shape[1]
    y_shape = grid_shape[0]
    x_range = tf.range(0, x_shape)
    y_range = tf.range(0, y_shape)

    grid_x = tf.tile(tf.reshape(x_range, [1, -1, 1, 1]), [y_shape, 1, 1, 1])
    grid_y = tf.tile(tf.reshape(y_range, [-1, 1, 1, 1]), [1, x_shape, 1, 1])
    grid = tf.cast(tf.concat([grid_x, grid_y], -1), feats.dtype)

    feats = tf.reshape(feats, [-1, y_shape, x_shape, nas, num_classes + 5])

    t_grid = make_tensor(grid_shape[::-1], ttype=feats.dtype)
    t_input = make_tensor(input_shape[::-1], ttype=feats.dtype)

    # Adjust predictions to each spatial grid point and anchor size.
    box_xy = (tf.sigmoid(feats[..., :2]) + grid) / t_grid
    box_wh = tf.exp(feats[..., 2:4]) * at / t_input
    box_confidence = tf.sigmoid(feats[..., 4:5])
    box_class_probs = tf.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    else:
        return box_xy, box_wh, box_confidence, box_class_probs

def correct_boxes(box_xy, box_wh, input_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    input_shape = make_tensor(input_shape, ttype=box_yx.dtype)
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)

    min_maxes = [box_mins[..., 0:1],   # y_min
                 box_mins[..., 1:2],   # x_min
                 box_maxes[..., 0:1],  # y_max
                 box_maxes[..., 1:2]]  # x_max

    boxes =  tf.concat(min_maxes, -1)
    boxes *= tf.concat([input_shape, input_shape], -1)

    return boxes

def boxes_and_scores(feats, anchors, nclasses, input_shape):
    xy, wh, conf, probs = yolo_head(feats, anchors, nclasses, input_shape)
    boxes = tf.reshape(correct_boxes(xy, wh, input_shape), [-1, 4])
    box_scores = tf.reshape(conf * probs, [-1, nclasses])

    return boxes, box_scores

def get_anchors(network):
    base = network['metadata']['base_image_network']
    anchors = network['metadata']['anchors']

    return [[anchors[idx] for idx in mask] for mask in MASKS[base]]

def output_boxes(outputs, anchors, nclasses, score_thresh=0.6, iou_thresh=0.5):
    input_shape = shape(outputs[0])[1:3] * 32

    boxes = []
    box_scores = []

    for out, ans in zip(outputs, anchors):
        bxs, scs = boxes_and_scores(out, ans, nclasses, input_shape)
        boxes.append(bxs)
        box_scores.append(scs)

    boxes = tf.concat(boxes, 0)
    box_scores = tf.concat(box_scores, 0)

    mask = box_scores >= score_thresh
    max_boxes_tensor = make_tensor(MAX_BOUNDING_BOXES, ttype='int32')

    boxes_ = []
    scores_ = []
    classes_ = []

    for c in range(nclasses):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(class_boxes,
                                                 class_box_scores,
                                                 max_boxes_tensor,
                                                 iou_threshold=iou_thresh)

        class_boxes = tf.gather(class_boxes, nms_index)
        class_box_scores = tf.gather(class_box_scores, nms_index)
        classes = tf.ones_like(class_box_scores, 'int32') * c

        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)

    boxes_ = tf.concat(boxes_, 0)
    scores_ = tf.concat(scores_, 0)
    classes_ = tf.concat(classes_, 0)

    return boxes_, scores_, classes_

def box_detector(cnn, readout, variables, nclasses, threshold):
    network = complete_image_network(cnn, readout)
    anchors = get_anchors(network)
    layers = network['layers']

    input_shape = graph_input_shape(network)
    X = tf.placeholder(tf.float32, shape=input_shape, name='input_data')
    Xin = normalize_image(X, network)

    trn = variables['is_training']
    keep_prob = variables['keep_prob']

    _, outputs = make_all_outputs(Xin, trn, layers[:-1], keep_prob)
    _, feats = yolo_output_branches(outputs, layers[-1], trn)

    box_preds = output_boxes(feats, anchors, nclasses, score_thresh=threshold)

    return {'bounding_box_X': X, 'bounding_box_preds': box_preds}

def box_projector(loader, variables, tf_session):
    X = variables['bounding_box_X']
    preds = variables['bounding_box_preds']
    is_training = variables['is_training']
    keep_prob = variables['keep_prob']

    batch_params = {keep_prob: 1.0, is_training: False}

    def boxes_for_image(image_path):
        batch_params[X] = np.expand_dims(loader(image_path), axis=0)
        return tf_session.run(preds, feed_dict=batch_params)

    return boxes_for_image
