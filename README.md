# BigML Sense/Net

Sense/Net is a BigML interface to Tensorflow, which takes a network
specification as a dictionary (read from BigML's JSON model format)
and instantiates a tensorflow compute graph based on that
specification.

## Entry Points

The library is meant, in general, to take a BigML model specification
and an optional map of settings and return a `tf.keras.Model` based on
these arguments.

### Deepnets

The class `sensenet.models.deepnet.DeepnetWrapper` exposes this
functionality for BigML deepnet models.  To instantiate one of these,
pass the model specification and the map of additional settings:

```
model = sensenet.models.deepnet.DeepnetWrapper(model_dict, settings)
```

Again, `model_dict` is typically the relevant section from the
downloaded BigML model, and `settings` is a map of optional settings
which may contain:

- `image_path_prefix`: A string directory indicating the path where
  images are to be found for image predictions.  When an image path is
  passed at prediction time, this string is prepended to the given
  path.

- `input_image_format`: The format of input images for the network.
  This can be either an image file on disk (`file`) or a string
  containing the raw image bytes (`bytes`)

Once instantiated, you can use the model to make predictions:

```
model.predict([1.0, 2.0, 3.0])
```

The input point or points must be a list (or nested list) containing
the input data for each point, in the order implied by
`model._preprocessors`.  Categorical and image variables should be
passed as strings, where the image is either a path to the image on
disk, or the raw compressed image bytes.

The function returns a numpy array where each row is the model's
prediction for each input point.  For classification models, there
will be a probability for each class in each row.  For regression
models, each row will contain only a single entry.
