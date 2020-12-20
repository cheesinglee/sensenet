# BigML Sense/Net

Sense/Net is a BigML interface to Tensorflow, which takes a network
specification as a dictionary (read from BigML's JSON model format)
and instantiates a TensorFlow compute graph based on that
specification.

## Entry Points

The library is meant, in general, to take a BigML model specification
as a JSON document, and an optional map of settings and return a
lightweight wrapper around a `tf.keras.Model` based on these
arguments.  The wrapper creation function can be found in
`sensenet.models.wrappers.create_model`

## Model Instantiation

To instantiate a model, pass the model specification and the dict
of additional, optional settings to `create_model`.  For example:

```
model = create_model(a_dict, settings={'image_path_prefix': 'images/path/'})
```

Again, `a_dict` is typically a downloaded BigML model, read into a
python dictionary via `json.load` or similar.

For image models, `settings` is either `None` or a dict of optional
settings which may contain:

- `bounding_box_threshold`: For object detection models only, the
  minimal score that an object can have and still be surfaced to the
  user as part of the output.  The default is 0.5, and lower the score
  will have the effect of more (possibly spurious) boxes identified in
  each input image.

- `color_space`: A string which is One of `['rgb', 'rgba', 'bgr',
  'bgra']`.  The first three letters give the order of the color
  channels (red, blue, and green) in the input tensors that will be
  passed to the model.  The final presense or absence of an `'a'`
  indicates that an alpha channel will be present (which will be
  ignored).  This can be useful to match the color space of the output
  model to that provided by another library, such as open CV.  Note
  that TensorFlow uses RGB ordering by default, and all files read by
  tensorflow are automatically read as RGB files.  This argument is
  generally only necessary if `input_image_format` is
  `'pixel_values'`, and will possibly break predictions if specified
  when the input is a file.

- `image_path_prefix`: A string directory indicating the path where
  images are to be found for image predictions.  When an image path is
  passed at prediction time, this string is prepended to the given
  path.

- `input_image_format`: The format of input images for the network.
  This can be either an image file on disk (`'file'`, the default), a
  string containing the raw, undecoded, image file bytes (`'bytes'`)
  or the decompressed image data represented as a nested python list,
  numpy array, or TensorFlow tensor of pixel values
  (`'pixel_values'`).

- `iou_threshold`: A threshold indicating the amount of overlap boxes
  predicting the same class should have before they are considered to
  be bounding the same object.  The default is 0.5, and lower values
  have the effect of eliminating boxes which would otherwise have been
  surfaced to the user.

- `max_objects`: The maximum number of bounding boxes to return for
  each image.  The default is 32.

- `rescale_type`: A string which is one of `['warp', 'pad', 'crop']`.
  If `'warp'`, input images are scaled to the input dimensions
  specified in the network, and their aspect ratios are *not*
  preserved.  If `'pad'`, the image is resized to the smallest
  dimensions such that the image fits into the input dimensions of the
  network, then padded with constant pixels either below or to the
  right to create an appropriately sized image.  For example, if the
  input dimesions of the network are 100 x 100, and we attempt to
  classify a 300 x 600 image, the image is first rescaled to 50 x 100
  (preserving its aspect ratio) then padded on the right to create a
  100 x 100 image.  If `'crop'`, the image is resized to the smallest
  dimension such that the input dimensions fit in the image, then the
  image is centrally cropped to make the specified sizes.  Using the
  sizes in previous example, the image would be rescaled to 100 x 200
  (preserving its aspect ratio) then cropped by 50 pixels on the top
  and bottom to create a 100 x 100 image.

## Usage

Once instantiated, you can use the model to make predictions by using
the returned model as a function, like so:

```
prediction = model([1.0, 2.0, 3.0])
```

The input point or points must be a list (or nested list) containing
the input data for each point, in the order implied by
`model._preprocessors`.  Categorical and image variables should be
passed as strings, where the image is either a path to the image on
disk, or the raw compressed image bytes.

For classification or regression models, the function returns a numpy
array where each row is the model's prediction for each input point.
For classification models, there will be a probability for each class
in each row.  For regression models, each row will contain only a
single entry.

For object detection models, the input should always be a single image
(again, either as a file path, compressed byte string, or an array of
pixel values, depending on the settings map, and the result will be
list of detected boxes, each one represented as a dictionary.  For
example:

```
In [5]: model('pizza_people.jpg')
Out[5]:
[{'box': [16, 317, 283, 414], 'label': 'pizza', 'score': 0.9726969599723816},
 {'box': [323, 274, 414, 332], 'label': 'pizza', 'score': 0.7364346981048584},
 {'box': [158, 29, 400, 327], 'label': 'person', 'score': 0.6204285025596619},
 {'box': [15, 34, 283, 336], 'label': 'person', 'score': 0.5346986055374146},
 {'box': [311, 23, 416, 255], 'label': 'person', 'score': 0.41961848735809326}]
```

The `box` array contains the coordinates of the detected box, as `x1,
y1, x2, y2`, where those coordinates represent the upper-left and
lower-right corners of each bounding box, in a coordinate system with
(0, 0) at the upper-left of the input image.  The `score` is the rough
probability that the object has been correctly identified, and the
`label` is the detected class of the object.
