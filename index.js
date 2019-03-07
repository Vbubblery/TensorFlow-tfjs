// Optional Load the binding:
// Use '@tensorflow/tfjs-node-gpu' if running with GPU.
const tf = require('@tensorflow/tfjs-node');

const input = tf.input({shape:[10]})

const layer1 = tf.layers.dense({units: 100, activation: 'relu'});
const layer2 = tf.layers.dense({units: 1, activation: 'linear'});

const output = layer2.apply(layer1.apply(input));

// Create the model based on the inputs.
const model = tf.model({inputs: input, outputs: output});
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

const xs = tf.randomNormal([100, 10]);
const ys = tf.randomNormal([100, 1]);

model.fit(xs, ys, {
  epochs: 100,
  callbacks: {
    onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
  }
});
