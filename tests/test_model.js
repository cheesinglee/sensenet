const bounding_boxes = false;
// uncomment if you're pointed at a bounding box model
// const bounding_boxes = true

require('@tensorflow/tfjs-node');

const { createCanvas, loadImage } = require('canvas');
const tf = require('@tensorflow/tfjs');

const model_file = 'file://data/test_model_save/model.json';
var image_file = null;

if (bounding_boxes)
    image_file = 'data/images/strange_car.png';
else
    image_file = 'data/images/dog.jpg';

var test_model = async function() {
    var model = await tf.loadGraphModel(model_file);
    const image = await loadImage(image_file);

    // There are slight differences prediction-wise between what we
    // get in python and what we get in JS.  I'm imagining the problem
    // is image loading, which is generally underspecified.  There's
    // no reason to believe what we have here is equivalent to what
    // happens in in TF in python.  These difference may be
    // unavoidable in general.
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext('2d');

    ctx.drawImage(image, 0, 0);

    // I had to jump through hoops to get an image to load locally in
    // node, but tf.browser.fromPixels will actually take an HTML
    // element as an argument, so this should be easier in the
    // browser.
    const img = tf.cast(tf.browser.fromPixels(canvas).expandDims(0), 'float32');

    if (bounding_boxes) {
        // When I do this, tf says that executeAsync is not necessary
        // here, but if I take it out and just do .predict, TF says
        // one of the ops returns a Promise and to use executeAsync.
        // I'm confused.
        const prediction = await model.executeAsync(img);

        var scores = await prediction[0].array();
        var classes = await prediction[1].array();
        var boxes = await prediction[2].array();

        console.assert(scores[0].length == 1);
        console.assert(classes[0].length == 1);
        console.assert(boxes[0].length == 1);

        console.assert(classes[0][0] == 2);
        console.assert(scores[0][0] > 0.95);

        var box = boxes[0][0];
        console.assert(550 < box[0]);
        console.assert(box[0] < 600);
        console.assert(220 < box[1]);
        console.assert(box[1] < 270);
        console.assert(970 < box[2]);
        console.assert(box[2] < 1020);
        console.assert(390 < box[3]);
        console.assert(box[3] < 440);
    }
    else {
        const prediction = await model.predict(img).array();

        for (var i = 0; i < prediction[0].length; i++) {
            if (i == 254) console.assert(prediction[0][i] > 0.7);
            else console.assert(prediction[0][i] < 0.02);
        }
    }

    console.log('Test complete.');
};

test_model();
