var layer_defs = [];
layer_defs.push({
  type: 'input',
  out_sx: 24,
  out_sy: 24,
  out_depth: 1
});
layer_defs.push({
  type: 'conv',
  sx: 5,
  filters: 8,
  stride: 1,
  pad: 2,
  activation: 'relu'
});
layer_defs.push({
  type: 'pool',
  sx: 2,
  stride: 2
});
layer_defs.push({
  type: 'conv',
  sx: 5,
  filters: 16,
  stride: 1,
  pad: 2,
  activation: 'relu'
});
layer_defs.push({
  type: 'pool',
  sx: 3,
  stride: 3
});
layer_defs.push({
  type: 'softmax',
  num_classes: 10
});

// create a net out of it
var net = new convnetjs.Net();
net.makeLayers(layer_defs);

// the network always works on Vol() elements. These are essentially
// simple wrappers around lists, but also contain gradients and dimensions
// line below will create a 1x1x2 volume and fill it with 0.5 and -1.3
// var x = new convnetjs.Vol([0.5, -1.3]);
// var y = 0
var trainer = new convnetjs.SGDTrainer(net, {
  method: 'adadelta',
  batch_size: 20,
  l2_decay: 0.001
});

var num_batches = 1; // 20 training batches, 1 test
var data_img_elts = new Array(num_batches);
var img_data = new Array(num_batches);
var loaded = new Array(num_batches);
var loaded_train_batches = [];


var sample_training_instance = function (b) {

  // find an unloaded batch

  // var b = 10; // the 10-th batch
  var k = Math.floor(Math.random() * 3000); // sample within the batch
  var n = b * 3000 + k;

  // fetch the appropriate row of the training image and reshape into a Vol
  var p = img_data[b].data;
  // console.log('k:' + k + '   n:' + n + '    length:' + labels.length +  '     length p:'+ p.length);
  var x = new convnetjs.Vol(28, 28, 1, 0.0);
  var W = 28 * 28;
  for (var i = 0; i < W; i++) {
    var ix = ((W * k) + i) * 4;
    x.w[i] = p[ix] / 255.0;
  }
  x = convnetjs.augment(x, 24);

  var isval = false;

  return {
    x: x,
    label: labels[n],
    isval: isval
  };
}


var visualize_activations = function (net, elt) {

  // clear the element
  elt.innerHTML = "";

  // show activations in each layer
  var N = net.layers.length;
  for (var i = 0; i < N; i++) {
    var L = net.layers[i];

    var layer_div = document.createElement('div');

    // visualize activations
    var activations_div = document.createElement('div');
    activations_div.appendChild(document.createTextNode('Activations:'));
    activations_div.appendChild(document.createElement('br'));
    activations_div.className = 'layer_act';
    var scale = 2;
    if (L.layer_type === 'softmax' || L.layer_type === 'fc') scale = 10; // for softmax
    draw_activations(activations_div, L.out_act, scale);

    layer_div.appendChild(activations_div);

    // print some stats on left of the layer
    layer_div.className = 'layer ' + 'lt' + L.layer_type;
    var title_div = document.createElement('div');
    title_div.className = 'ltitle'
    var t = L.layer_type + ' (' + L.out_sx + 'x' + L.out_sy + 'x' + L.out_depth + ')';
    title_div.appendChild(document.createTextNode(t));
    layer_div.appendChild(title_div);

    // css madness needed here...
    var clear = document.createElement('div');
    clear.className = 'clear';
    layer_div.appendChild(clear);

    elt.appendChild(layer_div);
  }
}


var draw_activations = function (elt, A, scale, grads) {

  var s = scale || 2; // scale
  var draw_grads = false;
  if (typeof (grads) !== 'undefined') draw_grads = grads;

  // get max and min activation to scale the maps automatically
  var w = draw_grads ? A.dw : A.w;
  var mm = maxmin(w);

  // create the canvas elements, draw and add to DOM
  for (var d = 0; d < A.depth; d++) {

    var canv = document.createElement('canvas');
    canv.className = 'actmap';
    var W = A.sx * s;
    var H = A.sy * s;
    canv.width = W;
    canv.height = H;
    var ctx = canv.getContext('2d');
    var g = ctx.createImageData(W, H);

    for (var x = 0; x < A.sx; x++) {
      for (var y = 0; y < A.sy; y++) {
        if (draw_grads) {
          var dval = Math.floor((A.get_grad(x, y, d) - mm.minv) / mm.dv * 255);
        } else {
          var dval = Math.floor((A.get(x, y, d) - mm.minv) / mm.dv * 255);
        }
        for (var dx = 0; dx < s; dx++) {
          for (var dy = 0; dy < s; dy++) {
            var pp = ((W * (y * s + dy)) + (dx + x * s)) * 4;
            for (var i = 0; i < 3; i++) {
              g.data[pp + i] = dval;
            } // rgb
            g.data[pp + 3] = 255; // alpha channel
          }
        }
      }
    }
    ctx.putImageData(g, 0, 0);
    elt.appendChild(canv);
  }
}


var load_data_batch = function (batch_num) {
  // Load the dataset with JS in background
  data_img_elts[batch_num] = new Image();
  var data_img_elt = data_img_elts[batch_num];

  function loadHandler() {
    var data_canvas = document.createElement('canvas');
    data_canvas.width = data_img_elt.width;
    data_canvas.height = data_img_elt.height;
    var data_ctx = data_canvas.getContext("2d");
    data_ctx.drawImage(data_img_elt, 0, 0); // copy it over... bit wasteful :(
    img_data[batch_num] = data_ctx.getImageData(0, 0, data_canvas.width, data_canvas.height);
    loaded[batch_num] = true;
    loaded_train_batches.push(batch_num);
    console.log('finished loading data batch ' + batch_num);
  };
  data_img_elt.onload = loadHandler;
  data_img_elt.src = "convnetjs/mnist/mnist_batch_" + batch_num + ".png";

  if (data_img_elt.complete) {
    loadHandler();
  }
}

var step_num = 0;
var lossGraph = new cnnvis.Graph();
var xLossWindow = new cnnutil.Window(100);
var wLossWindow = new cnnutil.Window(100);
var trainAccWindow = new cnnutil.Window(100);
var valAccWindow = new cnnutil.Window(100);
var maxmin = cnnutil.maxmin;
var f2t = cnnutil.f2t;

var paused = true;

function train_per() {

  if (paused) return;
  // load more batches over time
  if (step_num % 5000 === 0 && step_num > 0) {
    for (var i = 0; i < num_batches; i++) {
      if (!loaded[i]) {
        // load it
        load_data_batch(i);
        break; // okay for now
      }
    }
  }


  var bi = Math.floor(Math.random() * loaded_train_batches.length);
  var b = loaded_train_batches[bi];

  var sample = sample_training_instance(b);

  var x = sample.x;
  var y = sample.label;

  if (sample.isval) {
    // use x to build our estimate of validation error
    net.forward(x);
    var yhat = net.getPrediction();
    var val_acc = yhat === y ? 1.0 : 0.0;
    valAccWindow.add(val_acc);
    return; // get out
  }

  // train on it with network
  var stats = trainer.train(x, y);
  var lossx = stats.cost_loss;
  var lossw = stats.l2_decay_loss;

  // keep track of stats such as the average training error and loss
  var yhat = net.getPrediction();
  var train_acc = yhat === y ? 1.0 : 0.0;
  xLossWindow.add(lossx);
  wLossWindow.add(lossw);
  trainAccWindow.add(train_acc);

  // visualize training status
  var train_elt = document.getElementById("trainstats");
  train_elt.innerHTML = '';
  var t = 'Forward time per example: ' + stats.fwd_time + 'ms';
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  var t = 'Backprop time per example: ' + stats.bwd_time + 'ms';
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  var t = 'Classification loss: ' + f2t(xLossWindow.get_average());
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  var t = 'L2 Weight decay loss: ' + f2t(wLossWindow.get_average());
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  var t = 'Training accuracy: ' + f2t(trainAccWindow.get_average());
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));
  var t = 'Examples seen: ' + step_num;
  train_elt.appendChild(document.createTextNode(t));
  train_elt.appendChild(document.createElement('br'));


  // visualize activations
  if (step_num % 30 === 0) {
    var vis_elt = document.getElementById("visnet");
    visualize_activations(net, vis_elt);
    var cl = document.getElementById('cldiv');
    cl.innerHTML = 'label: ' + y;

    var probability_volume = net.forward(x);
    // console.log('probability that x is class 0: ' + probability_volume.w[0]);
    // prints 0.50101
    var d = document.getElementById('egdiv');
    d.innerHTML = 'probability that x is class 0: ' + probability_volume.w[0].toFixed(4);
  }

  // log progress to graph, (full loss)
  if (step_num % 200 === 0) {
    var xa = xLossWindow.get_average();
    var xw = wLossWindow.get_average();
    if (xa >= 0 && xw >= 0) { // if they are -1 it means not enough data was accumulated yet for estimates
      lossGraph.add(step_num, xa + xw);
      lossGraph.drawSelf(document.getElementById("lossgraph"));
    }
  }

  step_num++;


}

// user settings
var change_lr = function () {
  trainer.learning_rate = parseFloat(document.getElementById("lr_input").value);
  update_net_param_display();
}
var change_momentum = function () {
  trainer.momentum = parseFloat(document.getElementById("momentum_input").value);
  update_net_param_display();
}
var change_batch_size = function () {
  trainer.batch_size = parseFloat(document.getElementById("batch_size_input").value);
  update_net_param_display();
}
var change_decay = function () {
  trainer.l2_decay = parseFloat(document.getElementById("decay_input").value);
  update_net_param_display();
}

var toggle_pause = function () {
  paused = !paused;
  var btn = document.getElementById('buttontp');
  if (paused) {
    btn.value = 'Resume'
  } else {
    btn.value = 'Pause';
  }
}

var update_net_param_display = function () {
  document.getElementById('lr_input').value = trainer.learning_rate;
  document.getElementById('momentum_input').value = trainer.momentum;
  document.getElementById('batch_size_input').value = trainer.batch_size;
  document.getElementById('decay_input').value = trainer.l2_decay;
}

function start() {
  if (loaded[0]) {
    console.log('starting!');
    setInterval(train_per, 0); // lets go!
  } else {
    update_net_param_display();
    console.log('waiting!');
    load_data_batch(0);
    setTimeout(start, 1000);
  } // keep checking
}

function start_main() {

  for (var k = 0; k < loaded.length; k++) {
    loaded[k] = false;
  }
  // async load train set batch 0 (6 total train batches)
  start();
}