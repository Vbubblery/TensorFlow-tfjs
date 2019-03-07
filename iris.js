const tf = require('@tensorflow/tfjs-node');
const dataForge = require('data-forge');
require('data-forge-fs');

const df = dataForge
  .readFileSync('./iris.csv')
  .parseCSV()
  .parseFloats(["sepal.length","sepal.width","petal.length","petal.width"])

const df_X = df.subset(["sepal.length","sepal.width","petal.length"]).toRows();
const df_y = df.subset(["variety"]).toRows();
const train_X = tf.tensor(df_X);
const train_y = tf.tensor(df_y);
train_X.print()
