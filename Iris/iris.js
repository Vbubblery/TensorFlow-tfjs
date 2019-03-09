const tf = require('@tensorflow/tfjs-node');
const dataForge = require('data-forge');
require('data-forge-fs');

const df = dataForge
  .readFileSync('./iris.csv')
  .parseCSV()
  .parseFloats(["sepal.length","sepal.width","petal.length","petal.width"])

const df_X = df.subset(["sepal.length","sepal.width","petal.length"]).toRows();
// const df_y = df.subset(["variety"]).toRows();
const df_y = df.subset(["variety"]).select(row=>{
  let value;
  switch (row.variety) {
    case "Setosa":
      value = 0;
      break;
    case "Versicolor":
      value = 1;
      break;
    case "Virginica":
      value = 2;
      break;
    default:
      value = -1;
  }
  return {variety:value}
}).getSeries("variety").toArray();

const train_X = tf.tensor2d(df_X);
const train_y = tf.tensor1d(df_y,dtype='int32').oneHot(train_y, 3);

// train_y.print()

//
// const input = tf.input({shape:[4]})
// const layer1 = tf.layers.dense({units: 20, activation: 'relu'});
// const layer2 = tf.layers.dense({units: 40, activation: 'relu'});
// const layer3 = tf.layers.dense({units: 40, activation: 'relu'});
// const layer4 = tf.layers.dense({units: 20, activation: 'relu'});
// const layer5 = tf.layers.dense({units: 10, activation: 'relu'});
// const layer6 = tf.layers.dense({units: 3, activation: 'softmax'});
//
// // 调用dispose来清空已占用的 GPU 内存：
// train_X.dispose();
// train_y.dispose();
