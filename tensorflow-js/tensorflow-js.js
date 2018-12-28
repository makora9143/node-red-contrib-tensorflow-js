const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const when = require('when');
const { createCanvas, loadImage, Image } = require('canvas');

function createDenseModel() {
  const IMAGE_H = 28;
  const IMAGE_W = 28;
  const model = tf.sequential();
  model.add(tf.layers.flatten({inputShape: [IMAGE_H, IMAGE_W, 1]}));
  model.add(tf.layers.dense({units: 42, activation: 'relu'}));
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
  return model;
}

module.exports = function(RED) {
  function PredictNode(config) {
    RED.nodes.createNode(this,config);
    this.modelurl = config.modelurl;
    var node = this;
    tf.loadModel(node.modelurl).then(function (model) {
      node.on('input', function(msg) {
        const pred = model.predict(msg.input);
        let pred_idx = tf.argMax(pred, 1);
        pred_idx = pred_idx.dataSync();
        pred_idx = Array.from(pred_idx);
        msg.payload = pred_idx;
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
    this.validationSplit = 0.15;
    var node = this;
    node.on('input', function(msg) {
      const model = createDenseModel();
        msg.payload = msg.payload.toLowerCase();
        node.send(msg);
    });
  }
  function loadImage(src) {
    return when.promise(function(resolve, reject) {
      const img = new Image();
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
      // loadImage(msg.payload).then(function(img) {
        const canvas = createCanvas(img.naturalWidth, img.naturalHeight);
        const ctx = canvas.getContext('2d');
        console.log(canvas.width, img.naturalWidth);
        console.log(canvas.height, img.naturalHeight);
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

  function ClassLabelNode(config) {
    RED.nodes.createNode(this, config);
    this.classdata = JSON.parse(config.classdata);
    let node = this;
    node.on('input', function(msg) {
      msg.payload = node.classdata[msg.payload[0]];
      node.send({payload: msg.payload, res: msg.res, req:msg.req});
    });
  }

  RED.nodes.registerType("predict", PredictNode);
  RED.nodes.registerType("train", TrainNode);
  RED.nodes.registerType("dataload", DataloadNode);
  RED.nodes.registerType("img2tensor", Img2TensorNode);
  RED.nodes.registerType("classlabel", ClassLabelNode);
};
