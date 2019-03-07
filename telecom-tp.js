// const tf = require('@tensorflow/tfjs-node');
const dataForge = require('data-forge');
require('data-forge-fs');

const df = dataForge.readFileSync('./train.csv').parseCSV();

console.log(df.head(3).toRows());
