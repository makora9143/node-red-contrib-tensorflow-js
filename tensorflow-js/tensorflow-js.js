const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const when = require('when');
const { createCanvas, loadImage, Image } = require('canvas');

function createDenseModel(xTrain) {
  const model = tf.sequential();
  model.add(tf.layers.dense(
      {units: 10, activation: 'sigmoid', inputShape: [xTrain.shape[1]]}));
  model.add(tf.layers.dense({units: 3, activation: 'softmax'}));
  return model;
}

module.exports = function(RED) {
  function PredictNode(config) {
    RED.nodes.createNode(this,config);
    this.modelurl = config.modelurl;
    var node = this;
    tf.loadModel(node.modelurl).then(function (model) {
      node.on('input', function(msg) {
        msg.payload = model.predict(msg.input);
        node.send(msg);
      });
    });
  }

  function TrainNode(config) {
    RED.nodes.createNode(this,config);
    this.lr = config.lr;
    this.batchsize = config.batchsize;
    this.optimizer = config.optimzier;
    this.epochs = config.epochs;
    var node = this;
    node.on('input', function(msg) {
      console.log('Training model... Please wait.');
      const xTrain = msg.payload.xTrain;
      const yTrain = msg.payload.yTrain;
      const xTest = msg.payload.xTest;
      const yTest = msg.payload.yTest;
      const model = createDenseModel(xTrain);
      model.summary();

      let optimizer;
      if (node.optimizer == 'sgd') {
        optimizer = tf.train.sgd(node.lr);
      } else {
        optimizer = tf.train.adam(node.lr);
      }
      model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
      });

      model.fit(xTrain, yTrain, {
        epochs: node.epochs,
        validationData: [xTest, yTest],
      }).then(function () {
        console.log('done');
      });
      msg.payload = model;

      node.send(msg);
    });
  }
  function loadImage(src) {
    return when.promise(function(resolve, reject) {
      const img = new Image();
      img.crossOrigin = '';
      img.onload = () => resolve(img);
      img.onerror = (e) => reject(e);
      img.src = src;
    });
  }
  function DataloadNode(config) {
    RED.nodes.createNode(this,config);
    var node = this;
    node.on('input', function(msg) {
      const files = msg.req.files;
      loadImage(files[0].buffer).then(function(img) {
        const canvas = createCanvas(img.naturalWidth, img.naturalHeight);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        msg.canvas = canvas;
        node.send(msg);
      });
    });
  }
  function preprocessing(img_data, sizew=224, sizeh=224) {
    console.log('preprocessing');
    input = tf.fromPixels(img_data);
    input = tf.image.resizeBilinear(input, [sizew, sizeh]);
    input = tf.expandDims(input, 0);
    input = tf.div(input, tf.scalar(255));
    return input;
  }

  function Img2TensorNode(config) {
    RED.nodes.createNode(this,config);
    this.sizew = config.sizew;
    this.sizeh = config.sizeh;
    let node = this;
    node.on('input', function(msg) {
      msg.input = preprocessing(msg.canvas);
      node.send(msg);
    });
  }

  function ClassifyNode(config) {
    RED.nodes.createNode(this, config);
    this.classdata = JSON.parse(config.classdata);
    let node = this;
    node.on('input', function(msg) {
      const pred = msg.payload;
      let pred_idx = tf.argMax(pred, 1);
      pred_idx = pred_idx.dataSync();
      pred_idx = Array.from(pred_idx);
      msg.payload = node.classdata[pred_idx[0]];
      node.send({payload: msg.payload, res: msg.res, req:msg.req});
    });
  }
  function convertToTensors(data, targets, testSplit) {
    const numExamples = data.length;
    if (numExamples !== targets.length) {
      throw new Error('data and split have different numbers of examples');
    }

    // Randomly shuffle `data` and `targets`.
    const indices = [];
    for (let i = 0; i < numExamples; ++i) {
      indices.push(i);
    }
    tf.util.shuffle(indices);

    const shuffledData = [];
    const shuffledTargets = [];
    for (let i = 0; i < numExamples; ++i) {
      shuffledData.push(data[indices[i]]);
      shuffledTargets.push(targets[indices[i]]);
    }

    // Split the data into a training set and a tet set, based on `testSplit`.
    const numTestExamples = Math.round(numExamples * testSplit);
    const numTrainExamples = numExamples - numTestExamples;

    const xDims = shuffledData[0].length;

    // Create a 2D `tf.Tensor` to hold the feature data.
    const xs = tf.tensor2d(shuffledData, [numExamples, xDims]);

    // Create a 1D `tf.Tensor` to hold the labels, and convert the number label
    // from the set {0, 1, 2} into one-hot encoding (.e.g., 0 --> [1, 0, 0]).
    const ys = tf.oneHot(tf.tensor1d(shuffledTargets).toInt(), 3);

    // Split the data into training and test sets, using `slice`.
    const xTrain = xs.slice([0, 0], [numTrainExamples, xDims]);
    const xTest = xs.slice([numTrainExamples, 0], [numTestExamples, xDims]);
    const yTrain = ys.slice([0, 0], [numTrainExamples, 3]);
    const yTest = ys.slice([0, 0], [numTestExamples, 3]);

    return [xTrain, yTrain, xTest, yTest];
  }
  function getIrisData(iris_data){
    const IRIS_CLASSES = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'];
    return tf.tidy(() => {
      const dataByClass = [];
      const targetsByClass = [];
      for (let i = 0; i<IRIS_CLASSES.length; ++i) {
        dataByClass.push([]);
        targetsByClass.push([]);
      }
      for (const example of iris_data) {
        const target = Number(example.label);
        const data = Array(Number(example.f1), Number(example.f2), Number(example.f3), Number(example.f4));
        dataByClass[target].push(data);
        targetsByClass[target].push(target);
      }
      const xTrains = [];
      const yTrains = [];
      const xTests = [];
      const yTests = [];
      for (let i = 0; i < IRIS_CLASSES.length; ++i) {
        const [xTrain, yTrain, xTest, yTest] =
                convertToTensors(dataByClass[i], targetsByClass[i], 0.15);
        xTrains.push(xTrain);
        yTrains.push(yTrain);
        xTests.push(xTest);
        yTests.push(yTest);
      }
      const concatAxis = 0;
      return [
        tf.concat(xTrains, concatAxis), tf.concat(yTrains, concatAxis),
        tf.concat(xTests, concatAxis), tf.concat(yTests, concatAxis)
      ];
    });
  }
  function DatasetNode(config){
    RED.nodes.createNode(this, config);
    let node = this;
    node.on('input', function(msg) {
      const [xTrain, yTrain, xTest, yTest] = getIrisData(msg.payload);
      msg.payload = {xTrain: xTrain,
                     yTrain: yTrain,
                     xTest: xTest,
                     yTest: yTest};
      node.send(msg);
    });
  }

  RED.nodes.registerType("predict", PredictNode);
  RED.nodes.registerType("train", TrainNode);
  RED.nodes.registerType("dataload", DataloadNode);
  RED.nodes.registerType("img2tensor", Img2TensorNode);
  RED.nodes.registerType("classify", ClassifyNode);
  RED.nodes.registerType("dataset", DatasetNode);
};
