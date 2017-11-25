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


const DATASETS_CONFIG_JSON = 'deeplearnjs/gan_eval/model-builder-datasets-config.json';

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
var graph;
var session;
var critOptimizer;

var xTensor;

var critLoss;
var generatedImage;

var datasetDownloaded;
var datasetNames;
var selectedEnvName;
var selectedDatasetName;

var selectedModelName;
var genSelectedModelName;

var critSelectedOptimizerName;

var genLoadedWeights;
var dataSets;
var dataSet;
var xhrDatasetConfigs;

var critLearningRate;

var critMomentum;
var critNeedMomentum;
var critGamma;
var critBeta1;
var critBeta2;
var critNeedGamma;
var critNeedBeta;
var batchSize;

// var inputNDArrayVisualizers;
var fakeInputNDArrayVisualizers;

var inputShape;
var labelShape;
var randVectorShape;
var evalExamplesPerSec;
var examplesEvaluated;

var math;
// Keep one instance of each NDArrayMath so we don't create a user-initiated
// number of NDArrayMathGPU's.
var mathGPU = new NDArrayMathGPU();;
var mathCPU = new NDArrayMathCPU();;

function getImageDataOnly() {
    const [images, labels] = dataSet.getData();
    return images
}

function createOptimizer(which) {
    if (which === 'gen') {
        var selectedOptimizerName = genSelectedOptimizerName;
        var learningRate = genLearningRate;
        var momentum = genMomentum;
        var gamma = genGamma;
        var beta1 = genBeta1;
        var beta2 = genBeta2;
        var varName = 'generator';
    } else if (which === 'disc') {

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


function populateDatasets() {
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
            selectedDatasetName = datasetNames[0];
            xhrDatasetConfigs = _xhrDatasetConfigs;
            updateSelectedDataset(datasetNames[0]);
        },
        error => {
            throw new Error('Dataset config could not be loaded: ' + error);
        });
}

function updateSelectedDataset(datasetName) {

    graphRunner.stopEvaluating();
    graphRunner.stopInferring();

    if (dataSet != null) {
        dataSet.dispose();
    }

    selectedDatasetName = datasetName;
    // selectedModelName = '';
    // genSelectedModelName = '';
    // critSelectedModelName = '';
    dataSet = dataSets[datasetName];
    datasetDownloaded = false;
    // showDatasetStats = false;

    inputShape = dataSet.getDataShape(IMAGE_DATA_INDEX);
    //labelShape = dataSet.getDataShape(LABEL_DATA_INDEX);
    labelShape = [2]; // for gan there will be only two classes: real and fake
    randVectorShape = [100];

    // buildRealImageContainer();
    fakeInputNDArrayVisualizers = [];
    buildFakeImageContainer(document.querySelector('#generated-container'), fakeInputNDArrayVisualizers);

    dataSet.fetchData().then(() => {
        datasetDownloaded = true;
        dataSet.normalizeWithinBounds(IMAGE_DATA_INDEX, -1, 1);

        populateModelDropdown();
    });


}


function populateModelDropdown() {

    const _genModelNames = ['Custom'];
    const _critModelNames = ['Custom'];

    const modelConfigs =
        xhrDatasetConfigs[selectedDatasetName].modelConfigs;
    for (const modelName in modelConfigs) {
        if (modelConfigs.hasOwnProperty(modelName)) {
            if (modelName.endsWith('(gen)')) {
                _genModelNames.push(modelName);
            } else {
                _critModelNames.push(modelName);
            }
        }
    }

    genSelectedModelName = _genModelNames[_genModelNames.length - 1];
    critSelectedModelName = _critModelNames[_critModelNames.length - 1];

    generatorNet = new Net('gen', randVectorShape, inputShape);
    criticNet = new Net('crit', inputShape, labelShape);

    function loadSuccessCallback(which) {

        which.layerParamChanged();

        which.validateNet();

        isValid = criticNet.isValid && generatorNet.isValid

        console.log(`${which.name}valid`, which.isValid, 'allvalid:', isValid)

        if (isValid) {
            createModel();
        }
    }

    loadNetFromPath(xhrDatasetConfigs[selectedDatasetName].modelConfigs[genSelectedModelName].path, generatorNet, loadSuccessCallback);
    loadNetFromPath(xhrDatasetConfigs[selectedDatasetName].modelConfigs[critSelectedModelName].path, criticNet, loadSuccessCallback);

}

var generatorNet = null;
var criticNet = null;
class Net { // gen or disc or critic
    constructor(name, _inputShape, _outputShape) {
        this.name = name;
        this._inputShape = _inputShape;
        this._outputShape = _outputShape;
        this.isValid = false;
        this.hiddenLayers = [];
    }

    addLayer() {

        const modelLayer = new ModelLayer(); //document.createElement('model-layer');

        const lastHiddenLayer = this.hiddenLayers[this.hiddenLayers.length - 1];
        const lastOutputShape = lastHiddenLayer != null ?
            lastHiddenLayer.getOutputShape() :
            randVectorShape;
        this.hiddenLayers.push(modelLayer);

        modelLayer.initialize(window, lastOutputShape);
        // layerParamChanged(which)

        return modelLayer;
    }

    layerParamChanged() {
        // Go through each of the model layers and propagate shapes.

        let lastOutputShape = this._inputShape;

        for (let i = 0; i < this.hiddenLayers.length; i++) {
            lastOutputShape = this.hiddenLayers[i].setInputShape(lastOutputShape);
        }
    }

    validateNet() {
        let valid = true;

        var HiddenLayers = this.hiddenLayers;
        var lastLayerOutputShape = this._outputShape;

        for (let i = 0; i < HiddenLayers.length; ++i) {
            valid = valid && HiddenLayers[i].isValid();
        }
        if (HiddenLayers.length > 0) {
            const lastLayer = HiddenLayers[HiddenLayers.length - 1];
            valid = valid &&
                util.arraysEqual(lastLayerOutputShape, lastLayer.getOutputShape()); // for gen ,  lastLayerOutputShape = inputShape, for critic, lastLayerOutputShape = labelShape
        }

        this.isValid = valid && (HiddenLayers.length > 0);
    }
}


// ----------- make those two into generic utility functions by separating global variables outside and using callback for async return ----------------


function loadNetFromPath(modelPath, which, loadSuccessCallback) {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', modelPath);

    xhr.onload = () => {
        loadNetFromJson(xhr.responseText, which);
        loadSuccessCallback(which);
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

    lastOutputShape = which._inputShape;

    hiddenLayers = which.hiddenLayers;

    const layerBuilders = JSON.parse(modelJson);
    for (let i = 0; i < layerBuilders.length; i++) {
        const modelLayer = which.addLayer();
        modelLayer.loadParamsFromLayerBuilder(lastOutputShape, layerBuilders[i]);
        lastOutputShape = hiddenLayers[i].setInputShape(lastOutputShape);

    }
    // isValid = validateNet(criticNet.hiddenLayers, labelShape) && validateNet(generatorNet.hiddenLayers, inputShape);
}

function createModel() {
    if (session != null) {
        session.dispose();
    }

    modelInitialized = false;
    if (isValid === false) {
        return;
    }

    // Construct graph
    graph = new Graph();
    const g = graph;
    randomTensor = g.placeholder('random', randVectorShape);
    xTensor = g.placeholder('input', inputShape);
    oneTensor = g.placeholder('one', [2]);
    zeroTensor = g.placeholder('zero', [2]);

    const varianceInitializer = new VarianceScalingInitializer()
    const zerosInitializer = new ZerosInitializer()
    const onesInitializer = new OnesInitializer();

    // Construct generator
    let gen = randomTensor;
    for (let i = 0; i < generatorNet.hiddenLayers.length; i++) {
        let weights = null;
        if (genLoadedWeights != null) {
            weights = genLoadedWeights[i];
        }
        [gen] = generatorNet.hiddenLayers[i].addLayerMultiple(g, [gen],
            'generator', weights);
    }
    gen = g.tanh(gen);

    // Construct critic
    let crit1 = gen;
    let crit2 = xTensor; // real image
    for (let i = 0; i < criticNet.hiddenLayers.length; i++) {
        let weights = null;
        // if (loadedWeights != null) {
        //     weights = loadedWeights[i];
        // } // always need to retrain critic (which is the process of eval), never load weights for critic
        [crit1, crit2] = criticNet.hiddenLayers[i].addLayerMultiple(g, [crit1, crit2],
            'critic', weights);
    }

    generatedImage = gen;

    critPredictionReal = crit2;
    critPredictionFake = crit1;

    const critLossReal = g.softmaxCrossEntropyCost(
        critPredictionReal,
        oneTensor
    );
    const critLossFake = g.softmaxCrossEntropyCost(
        critPredictionFake,
        zeroTensor
    );
    critLoss = g.add(critLossReal, critLossFake); // js loss

    session = new Session(g, math);
    graphRunner.setSession(session);

    // startInference();

    modelInitialized = true;

    console.log('model initialized = true');
}



// --------------------  display and control  -------------------------------



function uploadWeights() {
    (document.querySelector('#weights-file')).click();
}

function setupUploadWeightsButton() {
    // Show and setup the load view button.
    const fileInput = document.querySelector('#weights-file');
    fileInput.addEventListener('change', event => {
        const file = fileInput.files[0];
        // Clear out the value of the file chooser. This ensures that if the user
        // selects the same file, we'll re-read it.
        fileInput.value = '';
        const fileReader = new FileReader();
        fileReader.onload = (evt) => {
            const weightsJson = fileReader.result;
            loadWeightsFromJson(weightsJson, which);
            createModel();
        };
        fileReader.readAsText(file);
    });
}

function loadWeightsFromJson(weightsJson, which) {
    genloadedWeights = JSON.parse(weightsJson);
}

function buildFakeImageContainer(inferenceContainer, fakeInputNDArrayVisualizers) {
    // const inferenceContainer =
    //     document.querySelector('#generated-container');
    inferenceContainer.innerHTML = '';
    // fakeInputNDArrayVisualizers = [];
    // fakeOutputNDArrayVisualizers = [];
    for (let i = 0; i < INFERENCE_EXAMPLE_COUNT; i++) {

        if (i % INFERENCE_EXAMPLE_ROWS === 0 && i !== 0) {
            linebreak = document.createElement("br");
            inferenceContainer.appendChild(linebreak);
        }

        const inferenceExampleElement = document.createElement('div');
        inferenceExampleElement.className = 'inference-example';
        inferenceExampleElement.style.display = 'inline';

        // Set up the input visualizer.
        const ndarrayImageVisualizer = new NDArrayImageVisualizer(inferenceExampleElement)

        ndarrayImageVisualizer.setShape(inputShape);
        ndarrayImageVisualizer.setSize(
            INFERENCE_IMAGE_SIZE_PX, INFERENCE_IMAGE_SIZE_PX);
        fakeInputNDArrayVisualizers.push(ndarrayImageVisualizer);

        inferenceContainer.appendChild(inferenceExampleElement);
    }
}

function displayBatchesEvaluated(totalBatchesEvaluated) {
    examplesEvaluated = batchSize * totalBatchesEvaluated;
    document.getElementById("examplesEvaluated").innerHTML = `Examples evaluated: ${examplesEvaluated}`
}

function displayEvalExamplesPerSec(_examplesPerSec) {

    evalExamplesPerSec =
        smoothExamplesPerSec(evalExamplesPerSec, _examplesPerSec);

    document.getElementById("evalExamplesPerSec").innerHTML = `Examples/sec: ${evalExamplesPerSec}`;
}


function smoothExamplesPerSec(
    lastExamplesPerSec, nextExamplesPerSec) {
    return Number((EXAMPLE_SEC_STAT_SMOOTHING_FACTOR * lastExamplesPerSec +
            (1 - EXAMPLE_SEC_STAT_SMOOTHING_FACTOR) * nextExamplesPerSec)
        .toPrecision(3));
}

function displayInferenceExamplesOutput(
    inputFeeds, inferenceOutputs) {

    // let realImages = [];
    let fakeImages = [];

    for (let i = 0; i < inferenceOutputs.length; i++) {
        // realImages.push(inputFeeds[i][0].data);
        fakeImages.push((inferenceOutputs[i]));

    }

    // realImages =
    //     dataSet.unnormalizeExamples(realImages, IMAGE_DATA_INDEX);

    fakeImages =
        dataSet.unnormalizeExamples(fakeImages, IMAGE_DATA_INDEX);

    // Draw the images.
    for (let i = 0; i < inferenceOutputs.length; i++) {
        // inputNDArrayVisualizers[i].saveImageDataFromNDArray(realImages[i]);
        fakeInputNDArrayVisualizers[i].saveImageDataFromNDArray(fakeImages[i]);
    }

    // Draw the logits.
    for (let i = 0; i < inferenceOutputs.length; i++) {

        // inputNDArrayVisualizers[i].draw();
        fakeInputNDArrayVisualizers[i].draw();

    }
}


var critLossGraph = new cnnvis.Graph();
var critLossWindow = new cnnutil.Window(200);

function displayCost(avgCost, which) {

    if (which === 'disc') {

    } else if (which === 'crit') {
        var cost = avgCost.get();
        var batchesEvaluated = graphRunner.getTotalBatchesEvaluated();

        critLossWindow.add(cost);

        var xa = critLossWindow.get_average();

        if (xa >= 0) { // if they are -1 it means not enough data was accumulated yet for estimates
            critLossGraph.add(batchesEvaluated, xa);
            critLossGraph.drawSelf(document.getElementById("critlossgraph"));
        }
    } else {

        // displayAccuracy(avgCost)
    }

}

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
    batchSize = 15;

    var envDropdown = document.getElementById("environment-dropdown");
    selectedEnvName = 'GPU';
    var ind = indexOfDropdownOptions(envDropdown.options, selectedEnvName)
    envDropdown.options[ind].selected = 'selected';
    updateSelectedEnvironment(selectedEnvName, graphRunner);

    // Default optimizer is momentum
    critSelectedOptimizerName = "adam";

    const eventObserver = {

        batchesEvaluatedCallback: (batchesEvaluated) =>
            displayBatchesEvaluated(batchesEvaluated),
        critCostCallback: (cost) => displayCost(cost, 'crit'),
        inferenceExamplesCallback:
            (inputFeeds, inferenceOutputs) =>
            displayInferenceExamplesOutput(inputFeeds, inferenceOutputs),
        evalExamplesPerSecCallback: (examplesPerSec) =>
            displayEvalExamplesPerSec(examplesPerSec),
        evalTotalTimeCallback: (totalTimeSec) => {
            totalTimeSec = totalTimeSec.toFixed(1);
            document.getElementById("evalTotalTimeSec").innerHTML = `Eval Total time: ${totalTimeSec} sec.`;
        },
    };
    graphRunner = new MyGraphRunner(math, session, eventObserver); // can do both inference and evaluate
    // graphRunner = new ImageEvalGraphRunner(math, session, eventObserver); // can inference but can't do evaluate

    genLoadedWeights = null;
    modelInitialized = false;

    /*

        const uploadModelButton = document.querySelector('#upload-model');
        uploadModelButton.addEventListener('click', () => uploadModel());
        setupUploadModelButton();
    
        const uploadWeightsButton = document.querySelector('#upload-weights');
        uploadWeightsButton.addEventListener('click', () => uploadWeights());
        setupUploadWeightsButton();
        */

    document.querySelector('#environment-dropdown').addEventListener('change', (event) => {
        selectedEnvName = event.target.value;
        updateSelectedEnvironment(selectedEnvName, graphRunner)
    });
    // critHiddenLayers = [];
    // genHiddenLayers = [];
    evalExamplesPerSec = 0;


    // Set up datasets.
    populateDatasets();
}

function monitor() {

    if (modelInitialized == false) {

        btn_infer.disabled = true;
        btn_infer.value = 'Initializing Model ...'
        // btn_train.disabled = true;
        // btn_train.style.visibility = 'hidden';
        btn_eval.style.visibility = 'hidden';

    } else {
        if (isValid) {

            btn_infer.disabled = false;
            // btn_train.style.visibility = 'visible';
            // Before clicking the eval button, first train the model for a while or load a pre-trained model to evaluate its samples against real images.Evaluate real images against real images not implemented yet.
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
                startInference();
            }

            if (eval_request) {
                eval_request = false;
                // createModel();
                startEvalulating();
            }

        } else {
            btn_infer.className = 'btn btn-danger btn-md';
            btn_infer.disabled = true;
            btn_infer.value = 'Model not valid'
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