/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

function getRandomInputProvider(shape) {
    return {
        getNextCopy(math) {
            return NDArray.randNormal(shape);
        },
        disposeCopy(math, copy) {
            copy.dispose();
        }
    }
}

function getDisplayShape(shape) {
    return `[${shape}]`;
}

var dl = deeplearn;
var Array1D = dl.Array1D;
var Array3D = dl.Array3D;
var DataStats = dl.DataStats;
var FeedEntry = dl.FeedEntry;
var Graph = dl.Graph;
var InCPUMemoryShuffledInputProviderBuilder = dl.InCPUMemoryShuffledInputProviderBuilder;
var Initializer = dl.Initializer;
var InMemoryDataset = dl.InMemoryDataset;
var MetricReduction = dl.MetricReduction;
// var MomentumOptimizer = dl.MomentumOptimizer;
// var SGDOptimizer = dl.SGDOptimizer;
// var RMSPropOptimizer = dl.RMSPropOptimizer;
// var AdagradOptimizer = dl.AdagradOptimizer;
// var AdadeltaOptimizer = dl.AdadeltaOptimizer;
var AdamOptimizer = dl.AdamOptimizer;
// var AdamaxOptimizer = dl.AdamaxOptimizer;
var NDArray = dl.NDArray;
var NDArrayMath = dl.NDArrayMath;
var NDArrayMathCPU = dl.NDArrayMathCPU;
var NDArrayMathGPU = dl.NDArrayMathGPU;
var Optimizer = dl.Optimizer;
var OnesInitializer = dl.OnesInitializer;
var Scalar = dl.Scalar;
var Session = dl.Session;
var Tensor = dl.Tensor;
var util = dl.util;
var VarianceScalingInitializer = dl.VarianceScalingInitializer;
var xhr_dataset = dl.xhr_dataset;
var XhrDataset = dl.XhrDataset;
var XhrDatasetConfig = dl.XhrDatasetConfig;
var ZerosInitializer = dl.ZerosInitializer;


const DATASETS_CONFIG_JSON = 'deeplearnjs/sample_eval/model-builder-datasets-config.json';

/** How often to evaluate the model against test data. */
const EVAL_INTERVAL_MS = 1500;
/** How often to compute the cost. Downloading the cost stalls the GPU. */
const COST_INTERVAL_MS = 500;
/** How many inference examples to show when evaluating accuracy. */
const INFERENCE_EXAMPLE_COUNT = 9; // must be a square number
const INFERENCE_EXAMPLE_ROWS = Math.sqrt(INFERENCE_EXAMPLE_COUNT);
const INFERENCE_IMAGE_SIZE_PX = 50;
/**
 * How often to show inference examples. This should be less often than
 * EVAL_INTERVAL_MS as we only show inference examples during an eval.
 */
const INFERENCE_EXAMPLE_INTERVAL_MS = 3000;

// Smoothing factor for the examples/s standalone text statistic.
const EXAMPLE_SEC_STAT_SMOOTHING_FACTOR = .7;

const TRAIN_TEST_RATIO = 5 / 6;

const IMAGE_DATA_INDEX = 0;
const LABEL_DATA_INDEX = 1;

var isValid;
var applicationState;
var modelInitialized;

var graphRunner;
var session;

var datasetDownloaded;
var datasetNames;
var selectedEnvName;
// var selectedDatasetName;

var selectedModelName;
var genSelectedModelName;

var critSelectedOptimizerName;

var dataSets;
var dataSet;
// var xhrDatasetConfigs;

var critLearningRate;

var critMomentum;
var critNeedMomentum;
var critGamma;
var critBeta1;
var critBeta2;
var critNeedGamma;
var critNeedBeta;
var batchSize;

var math;
// Keep one instance of each NDArrayMath so we don't create a user-initiated
// number of NDArrayMathGPU's.
var mathGPU = new NDArrayMathGPU();;
var mathCPU = new NDArrayMathCPU();;

function getImageDataOnly() {
    const [images, labels] = dataSet.getData();
    return images
}

function createOptimizer(which, graph) {
    if (which === 'gen') {
        // var selectedOptimizerName = genSelectedOptimizerName;
        // var learningRate = genLearningRate;
        // var momentum = genMomentum;
        // var gamma = genGamma;
        // var beta1 = genBeta1;
        // var beta2 = genBeta2;
        // var varName = 'generator';
    } else { // critic
        var selectedOptimizerName = critSelectedOptimizerName;
        var learningRate = critLearningRate;
        var momentum = critMomentum;
        var gamma = critGamma;
        var beta1 = critBeta1;
        var beta2 = critBeta2;
        var varName = 'critic';
    }
    switch (selectedOptimizerName) {
        case 'sgd':
            {
                return new SGDOptimizer(+learningRate,
                    graph.getNodes().filter((x) =>
                        x.name.startsWith(varName)));
            }
        case 'momentum':
            {
                return new MomentumOptimizer(+learningRate, +momentum,
                    graph.getNodes().filter((x) =>
                        x.name.startsWith(varName)));
            }
        case 'rmsprop':
            {
                return new RMSPropOptimizer(+learningRate, +gamma,
                    graph.getNodes().filter((x) =>
                        x.name.startsWith(varName)));
            }
        case 'adagrad':
            {
                return new AdagradOptimizer(+learningRate,
                    graph.getNodes().filter((x) =>
                        x.name.startsWith(varName)));
            }
        case 'adadelta':
            {
                return new AdadeltaOptimizer(+learningRate, +gamma,
                    graph.getNodes().filter((x) =>
                        x.name.startsWith(varName)));
            }
        case 'adam':
            {
                return new AdamOptimizer(+learningRate, +beta1, +beta2,
                    graph.getNodes().filter((x) =>
                        x.name.startsWith(varName)));
            }
        default:
            {
                throw new Error(`Unknown optimizer`);
            }
    }
}

// ------------------------- build model -----------------------------

// this is a global function for preparing the datasets for all models within this application

function fetchConfig_DownloadData(fetchConfigCallback) {
    dataSets = {};
    xhr_dataset.getXhrDatasetConfig(DATASETS_CONFIG_JSON).then(
        _xhrDatasetConfigs => {

            for (const datasetName in _xhrDatasetConfigs) {
                if (_xhrDatasetConfigs.hasOwnProperty(datasetName)) {
                    dataSets[datasetName] =
                        new XhrDataset(_xhrDatasetConfigs[datasetName]);
                }
            }
            datasetNames = Object.keys(dataSets);
            selectedDatasetName = datasetNames[0]; // 0: MNIST,  1: FashionMNIST 2: CIFAR10

            dataSet = dataSets[selectedDatasetName];

            // inputShape = dataSet.getDataShape(IMAGE_DATA_INDEX);
            //labelShape = dataSet.getDataShape(LABEL_DATA_INDEX);

            fetchConfigCallback(_xhrDatasetConfigs, selectedDatasetName);

            datasetDownloaded = false;

            dataSet.fetchData().then(() => {
                dataSet.normalizeWithinBounds(IMAGE_DATA_INDEX, -1, 1);
                datasetDownloaded = true;
            });

        },
        error => {
            throw new Error('Dataset config could not be loaded: ' + error);
        });
}

// -------- global function to build all needed models within the application

var evalModel;

function buildModels(xhrDatasetConfigs, selectedDatasetName) {

    const modelConfigs = xhrDatasetConfigs[selectedDatasetName].modelConfigs;

    evalModel = new EvalSampleModel(modelConfigs);

    evalModel.initialize();

}

// refactor those two load functions into generic utility functions by separating  
// global variables outside and using callback for async return

function loadNetFromPath(modelPath, which) {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', modelPath);

    xhr.onload = () => {
        loadNetFromJson(xhr.responseText, which);
        // which.layerParamChanged()
        which.validateNet();

        isValid = evalModel.criticNet.isValid && evalModel.generatorNet.isValid;

        console.log(`${which.name}valid`, which.isValid, 'allvalid:', isValid);

        if (isValid) {
            evalModel.createModel();
        }

    };
    xhr.onerror = (error) => {
        throw new Error(
            'Model could not be fetched from ' + modelPath + ': ' + error);
    };
    xhr.send();
}

function loadNetFromJson(modelJson, which) {
    var lastOutputShape;
    var hiddenLayers;

    lastOutputShape = which.inputShape;

    hiddenLayers = which.hiddenLayers;

    const layerBuilders = JSON.parse(modelJson);
    for (let i = 0; i < layerBuilders.length; i++) {
        const modelLayer = which.addLayer();
        modelLayer.loadParamsFromLayerBuilder(lastOutputShape, layerBuilders[i]);
        lastOutputShape = hiddenLayers[i].setInputShape(lastOutputShape);

    }
}

// --------------------  display and control  -------------------------------

function updateSelectedEnvironment(selectedEnvName, _graphRunner = null) {
    math = (selectedEnvName === 'GPU') ? mathGPU : mathCPU;
    console.log('math =', math === mathGPU ? 'mathGPU' : 'mathCPU')
    if (_graphRunner != null) {
        _graphRunner.setMath(math);
    }
}

// user settings
var changeNetParam = function () {

    critLearningRate = parseFloat(document.getElementById("crit-learning-rate-input").value);
    if (graphRunner.critOptimizer != null && critLearningRate !== graphRunner.critOptimizer.learningRate) {
        graphRunner.critOptimizer.learningRate = critLearningRate;
        console.log('crit learning rate changed to' + critLearningRate);
    }

    batchSize = parseFloat(document.getElementById("batch_size_input").value);
    if (graphRunner.batchSize != null && batchSize != graphRunner.batchSize) {
        graphRunner.batchSize = batchSize;
        console.log('batch size changed to' + batchSize);
    }

    // updateNetParamDisplay();
}

var infer_request = null;
var btn_infer = document.getElementById('buttoninfer');
var infer_paused = true;
btn_infer.addEventListener('click', () => {

    infer_paused = !infer_paused;
    if (infer_paused) {
        btn_infer.value = 'Start Inferring';
        if (graphRunner != null) {
            graphRunner.stopInferring(); // can return quickly
        }
    } else {
        infer_request = true;
        // graphRunner.startInference(); // can't return quickly, so put it outside to be monitored
        btn_infer.value = 'Pause Inferring';
    }
});

var eval_request = null;
var btn_eval = document.getElementById('buttoneval1');
var eval_paused = true;
btn_eval.addEventListener('click', () => {
    eval_paused = !eval_paused;

    if (eval_paused) {
        if (graphRunner != null) {
            graphRunner.stopEvaluating(); // can return quickly
        }
        btn_eval.value = 'Start Evaluating';

    } else {

        eval_request = true;
        // graphRunner.startEvaluating(); // can't return quickly, so put it outside to be monitored
        btn_eval.value = 'Pause Evaluating';

    }
});

// ----------------------- application initialization ----------------------


function run() {

    critLearningRate = 0.01;
    critMomentum = 0.1;
    critNeedMomentum = false;
    critGamma = 0.1;
    critBeta1 = 0.9;
    critBeta2 = 0.999;
    critNeedGamma = false;
    critNeedBeta = false;
    batchSize = 30;

    // Default optimizer is momentum
    critSelectedOptimizerName = "adam";

    const eventObserver = {
        batchesEvaluatedCallback: (batchesEvaluated) =>
            evalModel.displayBatchesEvaluated(batchesEvaluated),

        critCostCallback: (cost) => {
            var batchesEvaluated = graphRunner.getTotalBatchesEvaluated();
            evalModel.displayCost(cost, batchesEvaluated)
        },

        inferenceExamplesCallback:
            (inputFeeds, inferenceOutputs) =>
            evalModel.displayInferenceExamplesOutput(inputFeeds, inferenceOutputs),

        evalExamplesPerSecCallback: (examplesPerSec) =>
            evalModel.displayEvalExamplesPerSec(examplesPerSec),
        // evalTotalTimeCallback: (totalTimeSec) => {
        //     totalTimeSec = totalTimeSec.toFixed(1);
        //     document.getElementById("evalTotalTimeSec").innerHTML = `Eval Total time: ${totalTimeSec} sec.`;
        // },
    };
    graphRunner = new MyGraphRunner(math, session, eventObserver); // can do both inference and evaluate
    // graphRunner = new ImageEvalGraphRunner(math, session, eventObserver); // can inference but can't do evaluate

    var envDropdown = document.getElementById("environment-dropdown");
    selectedEnvName = 'GPU';
    var ind = indexOfDropdownOptions(envDropdown.options, selectedEnvName)
    envDropdown.options[ind].selected = 'selected';
    updateSelectedEnvironment(selectedEnvName, graphRunner);

    modelInitialized = false;

    document.querySelector('#environment-dropdown').addEventListener('change', (event) => {
        selectedEnvName = event.target.value;
        updateSelectedEnvironment(selectedEnvName, graphRunner)
    });


    // Set up datasets.
    fetchConfig_DownloadData(buildModels);

}

function monitor() {

    if (datasetDownloaded == false) {
        btn_infer.disabled = true;
        btn_infer.value = 'Downloading data'
        btn_eval.style.visibility = 'hidden';

    } else {
        if (modelInitialized) {
            if (isValid) {

                btn_infer.disabled = false;
                // Before clicking the eval button, first load a pre-trained model to evaluate its samples against real images.Evaluate real images against real images not implemented yet.
                btn_eval.style.visibility = 'visible';

                if (infer_paused) {
                    btn_infer.value = 'Start Infering'
                } else {
                    btn_infer.value = 'Stop Infering'
                }

                if (eval_paused) {
                    btn_eval.value = 'Start Evaluating'
                } else {
                    btn_eval.value = 'Stop Evaluating'
                }

                if (infer_request) {
                    infer_request = false;
                    // createModel();
                    evalModel.startInference();
                }

                if (eval_request) {
                    eval_request = false;
                    // createModel();
                    evalModel.startEvalulating();
                }

            } else {
                btn_infer.className = 'btn btn-danger btn-md';
                btn_infer.disabled = true;
                btn_infer.value = 'Model not valid'
                btn_eval.style.visibility = 'hidden';
            }
        } else {
            btn_infer.disabled = true;
            btn_infer.value = 'Initializing Model ...'
            btn_eval.style.visibility = 'hidden';
        }
    }

    setTimeout(function () {
        monitor();
    }, 100);
}

function start() {

    supported = detect_support();

    if (supported) {
        console.log('device & webgl supported');
        btn_infer.disabled = false;
        // btn_train.disabled = false;
        btn_eval.disabled = false;

        setTimeout(function () {

            run(); // initialize data and model

            monitor(); // monitor button clicks and display update

        }, 0);

    } else {
        console.log('device/webgl not supported')
        btn_infer.disabled = true;
        // btn_train.disabled = true;
        btn_eval.disabled = true;
    }
}