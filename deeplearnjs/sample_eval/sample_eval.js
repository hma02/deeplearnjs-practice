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

var Normalization = {
    NORMALIZATION_NEGATIVE_ONE_TO_ONE: 0,
    NORMALIZATION_ZERO_TO_ONE: 1,
    NORMALIZATION_NONE: 2
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
var MomentumOptimizer = dl.MomentumOptimizer;
var SGDOptimizer = dl.SGDOptimizer;
var RMSPropOptimizer = dl.RMSPropOptimizer;
var AdagradOptimizer = dl.AdagradOptimizer;
var AdadeltaOptimizer = dl.AdadeltaOptimizer;
var AdamOptimizer = dl.AdamOptimizer;
var AdamaxOptimizer = dl.AdamaxOptimizer;
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
const INFERENCE_EXAMPLE_COUNT = 1;
const INFERENCE_IMAGE_SIZE_PX = 100;
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

var ApplicationState = {
    IDLE: 1,
    TRAINING: 2
};


var isValid;
var totalTimeSec;
var applicationState;
var modelInitialized;
var showTrainStats;
var selectedNormalizationOption = Normalization.NORMALIZATION_NEGATIVE_ONE_TO_ONE;

// Datasets and models.
var graphRunner;
var graph;
var session;
var critOptimizer;
var genOptimizer;
var xTensor;
var labelTensor;
var costTensor;
var accuracyTensor;
var predictionTensor;

var genLoss;
var critLoss;
var generatedImage;

var datasetDownloaded;
var datasetNames;
var selectedEnvName;
var selectedDatasetName;
var modelNames;
var genModelNames;
var selectedModelName;
var genSelectedModelName;
var optimizerNames;
var genSelectedOptimizerName;
var critSelectedOptimizerName;
var loadedWeights;
var genLoadedWeights;
var dataSets;
var dataSet;
var xhrDatasetConfigs;
var datasetStats;
var critLearningRate;
var genLearningRate;
var critMomentum;
var genMomentum;
var critNeedMomentum;
var genNeedMomentum;
var critGamma;
var genGamma;
var critBeta1;
var critBeta2;
var genBeta1;
var genBeta2;
var critNeedGamma;
var genNeedGamma;
var critNeedBeta;
var genNeedBeta;
var batchSize;

// Stats.
var showDatasetStats;
var statsInputRange;
var statsInputShapeDisplay;
var statsLabelShapeDisplay;
var statsExampleCount;

// Charts.
var costChart;
var accuracyChart;
var examplesPerSecChart;
var costChartData;
var accuracyChartData;
var examplesPerSecChartData;

var trainButton;

// Visualizers.
var inputNDArrayVisualizers;
var outputNDArrayVisualizers;

var inputShape;
var labelShape;
var randVectorShape;
var examplesPerSec;
var evalExamplesPerSec;
var examplesTrained;
var examplesEvaluated;
var inferencesPerSec;
var inferenceDuration;
var generationsPerSec;
var generationDuration;

var inputLayer;
var hiddenLayers;

var layersContainer;
var critLayersContainer;
var genHiddenLayers;
var critHiddenLayers;

var math;
// Keep one instance of each NDArrayMath so we don't create a user-initiated
// number of NDArrayMathGPU's.
var mathGPU = new NDArrayMathGPU();;
var mathCPU = new NDArrayMathCPU();;


function getTestData() {
    const data = dataSet.getData();
    if (data == null) {
        return null;
    }
    const [images, labels] = dataSet.getData();

    const start = Math.floor(TRAIN_TEST_RATIO * images.length);

    return [images.slice(start), labels.slice(start)];
}

function getTrainingData() {
    const [images, labels] = dataSet.getData();

    const end = Math.floor(TRAIN_TEST_RATIO * images.length);

    return [images.slice(0, end), labels.slice(0, end)];
}

function getData() {
    return dataSet.getData();
}

function getImageDataOnly() {
    const [images, labels] = dataSet.getData();
    return images
}

function startInference() {
    const data = getImageDataOnly();
    if (data == null) {
        return;
    }
    if (isValid && (data != null)) {
        const shuffledInputProviderGenerator =
            new InCPUMemoryShuffledInputProviderBuilder([data]);
        const [inputImageProvider] =
        shuffledInputProviderGenerator.getInputProviders();

        const oneInputProvider = {
            getNextCopy(math) {
                return Array1D.new([0, 1]);
            },
            disposeCopy(math, copy) {
                copy.dispose();
            }
        }

        const zeroInputProvider = {
            getNextCopy(math) {
                return Array1D.new([1, 0]);
            },
            disposeCopy(math, copy) {
                copy.dispose();
            }
        }

        const inferenceFeeds = [{
                tensor: xTensor,
                data: inputImageProvider
            },
            {
                tensor: randomTensor,
                data: getRandomInputProvider(randVectorShape)
            },
            {
                tensor: oneTensor,
                data: oneInputProvider
            },
            {
                tensor: zeroTensor,
                data: zeroInputProvider
            }
        ]

        graphRunner.infer(
            generatedImage, null, null,
            inferenceFeeds, INFERENCE_EXAMPLE_INTERVAL_MS, INFERENCE_EXAMPLE_COUNT
        );
    }
}

function resetHyperParamRequirements(which) {
    if (which === 'gen') {
        genNeedMomentum = false;
        genNeedGamma = false;
        genNeedBeta = false;
    } else {
        critNeedMomentum = false;
        critNeedGamma = false;
        critNeedBeta = false;
    }
}

/**
 * Set flag to disable input by optimizer selection.
 */
function refreshHyperParamRequirements(optimizerName,
    which) {
    resetHyperParamRequirements(which);
    switch (optimizerName) {
        case "sgd":
            {
                // No additional hyper parameters
                break;
            }
        case "momentum":
            {
                if (which === 'gen') {
                    genNeedMomentum = true;
                } else {
                    critNeedMomentum = true;
                }
                break;
            }
        case "rmsprop":
            {
                if (which === 'gen') {
                    genNeedMomentum = true;
                    genNeedGamma = true;
                } else {
                    critNeedMomentum = true;
                    critNeedGamma = true;
                }
                break;
            }
        case "adagrad":
            {
                break;
            }
        case 'adadelta':
            {
                if (which === 'gen') {
                    genNeedGamma = true;
                } else {
                    critNeedGamma = true;
                }
                break;
            }
        case 'adam':
            {
                if (which === 'gen') {
                    genNeedBeta = true;
                } else {
                    critNeedBeta = true;
                }
                break;
            }
        case 'adamax':
            {
                if (which === 'gen') {
                    genNeedBeta = true;
                } else {
                    critNeedBeta = true;
                }
                break;
            }
        default:
            {
                throw new Error(`Unknown optimizer`);
            }
    }
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
    for (let i = 0; i < genHiddenLayers.length; i++) {
        let weights = null;
        if (loadedWeights != null) {
            weights = loadedWeights[i];
        }
        [gen] = genHiddenLayers[i].addLayerMultiple(g, [gen],
            'generator', weights);
    }
    gen = g.tanh(gen);

    // Construct critic
    let crit1 = gen;
    let crit2 = xTensor; // real image
    for (let i = 0; i < critHiddenLayers.length; i++) {
        let weights = null;
        // if (loadedWeights != null) {
        //     weights = loadedWeights[i];
        // } // always need to retrain critic (which is the process of eval), never load weights for critic
        [crit1, crit2] = critHiddenLayers[i].addLayerMultiple(g, [crit1, crit2],
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

function populateDatasets() {
    dataSets = {};
    xhr_dataset.getXhrDatasetConfig(DATASETS_CONFIG_JSON)
        .then(
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
    if (dataSet != null) {
        dataSet.removeNormalization(IMAGE_DATA_INDEX);
    }

    graphRunner.stopEvaluating();
    graphRunner.stopInferring();

    if (dataSet != null) {
        dataSet.dispose();
    }

    selectedDatasetName = datasetName;
    selectedModelName = '';
    dataSet = dataSets[datasetName];
    datasetDownloaded = false;
    showDatasetStats = false;

    dataSet.fetchData().then(() => {
        datasetDownloaded = true;
        applyNormalization(selectedNormalizationOption);
        setupDatasetStats();
        if (isValid) {
            createModel();
        }
        // Get prebuilt models.
        populateModelDropdown();
    });

    inputShape = dataSet.getDataShape(IMAGE_DATA_INDEX);
    //labelShape = dataSet.getDataShape(LABEL_DATA_INDEX);
    labelShape = [2];

    // critLayersContainer =
    //     document.querySelector('#crit-hidden-layers');

    // // CRITIC
    // critInputLayer = document.querySelector('#crit-input-layer');
    // insertLayerTableRow(critInputLayer, 'crit-input-layer', null, getDisplayShape(inputShape));

    // const critCostLayer = document.querySelector('#crit-cost-layer');
    // insertLayerTableRow(critCostLayer, 'crit-cost-layer', labelShapeDisplay, labelShapeDisplay);

    buildRealImageContainer();
    buildFakeImageContainer();
}

/* Helper function for building out container for images*/
function buildRealImageContainer() {
    const inferenceContainer =
        document.querySelector('#real-container');
    inferenceContainer.innerHTML = '';
    inputNDArrayVisualizers = [];
    outputNDArrayVisualizers = [];
    for (let i = 0; i < INFERENCE_EXAMPLE_COUNT; i++) {
        const inferenceExampleElement = document.createElement('div');
        inferenceExampleElement.className = 'inference-example';

        // Set up the input visualizer.

        const ndarrayImageVisualizer = new NDArrayImageVisualizer(inferenceExampleElement);
        ndarrayImageVisualizer.setShape(inputShape);
        ndarrayImageVisualizer.setSize(
            INFERENCE_IMAGE_SIZE_PX, INFERENCE_IMAGE_SIZE_PX);
        inputNDArrayVisualizers.push(ndarrayImageVisualizer);


        inferenceContainer.appendChild(inferenceExampleElement);
    }
}

function buildFakeImageContainer() {
    const inferenceContainer =
        document.querySelector('#generated-container');
    inferenceContainer.innerHTML = '';
    fakeInputNDArrayVisualizers = [];
    fakeOutputNDArrayVisualizers = [];
    for (let i = 0; i < INFERENCE_EXAMPLE_COUNT; i++) {
        const inferenceExampleElement = document.createElement('div');
        inferenceExampleElement.className = 'inference-example';

        // Set up the input visualizer.
        const ndarrayImageVisualizer = new NDArrayImageVisualizer(inferenceExampleElement)

        ndarrayImageVisualizer.setShape(inputShape);
        ndarrayImageVisualizer.setSize(
            INFERENCE_IMAGE_SIZE_PX, INFERENCE_IMAGE_SIZE_PX);
        fakeInputNDArrayVisualizers.push(ndarrayImageVisualizer);


        inferenceContainer.appendChild(inferenceExampleElement);
    }
}

function populateModelDropdown() {
    const _modelNames = ['Custom'];
    const _genModelNames = ['Custom'];
    const _critModelNames = ['Custom'];

    const modelConfigs =
        xhrDatasetConfigs[selectedDatasetName].modelConfigs;
    for (const modelName in modelConfigs) {
        if (modelConfigs.hasOwnProperty(modelName)) {
            if (modelName.endsWith('(disc)')) {

            } else if (modelName.endsWith('(gen)')) {
                _genModelNames.push(modelName);
            } else {
                _critModelNames.push(modelName);
            }
        }
    }

    genModelNames = _genModelNames;
    critModelNames = _critModelNames;
    genSelectedModelName = genModelNames[genModelNames.length - 1];
    critSelectedModelName = critModelNames[critModelNames.length - 1];
    updateSelectedModel(genSelectedModelName, 'gen');
    updateSelectedModel(critSelectedModelName, 'crit');
}

function updateSelectedModel(modelName, which) {
    removeAllLayers(which);
    if (modelName === 'Custom') {
        // TODO(nsthorat): Remember the custom layers.
        return;
    }

    loadModelFromPath(xhrDatasetConfigs[selectedDatasetName].modelConfigs[modelName].path, which);
}

function loadModelFromPath(modelPath, which) {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', modelPath);

    xhr.onload = () => {
        loadModelFromJson(xhr.responseText, which);
    };
    xhr.onerror = (error) => {
        throw new Error(
            'Model could not be fetched from ' + modelPath + ': ' + error);
    };
    xhr.send();
}

function setupDatasetStats() {
    // datasetStats = dataSet.getStats();
    // statsExampleCount = datasetStats[IMAGE_DATA_INDEX].exampleCount;
    // document.getElementById("statsExampleCount").innerHTML = `${statsExampleCount}`;
    // statsInputRange = '[' + datasetStats[IMAGE_DATA_INDEX].inputMin +
    //     ', ' + datasetStats[IMAGE_DATA_INDEX].inputMax + ']';
    // document.getElementById("statsInputRange").innerHTML = `${statsInputRange}`;
    // statsInputShapeDisplay = getDisplayShape(
    //     datasetStats[IMAGE_DATA_INDEX].shape);
    // document.getElementById("statsInputShapeDisplay").innerHTML = `${statsInputShapeDisplay}`;
    // statsLabelShapeDisplay = getDisplayShape(
    //     datasetStats[LABEL_DATA_INDEX].shape);
    // document.getElementById("statsLabelShapeDisplay").innerHTML = `${statsLabelShapeDisplay}`;
    // showDatasetStats = true;
}

function applyNormalization(selectedNormalizationOption) {
    switch (selectedNormalizationOption) {
        case Normalization.NORMALIZATION_NEGATIVE_ONE_TO_ONE:
            {
                dataSet.normalizeWithinBounds(IMAGE_DATA_INDEX, -1, 1);
                break;
            }
        case Normalization.NORMALIZATION_ZERO_TO_ONE:
            {
                dataSet.normalizeWithinBounds(IMAGE_DATA_INDEX, 0, 1);
                break;
            }
        case Normalization.NORMALIZATION_NONE:
            {
                dataSet.removeNormalization(IMAGE_DATA_INDEX);
                break;
            }
        default:
            {
                throw new Error('Normalization option must be 0, 1, or 2');
            }
    }
    setupDatasetStats();
}


function displayBatchesTrained(totalBatchesTrained) {
    examplesTrained = batchSize * totalBatchesTrained;
    document.getElementById("examplesTrained").innerHTML = `Examples trained: ${examplesTrained}`
}

function displayBatchesEvaluated(totalBatchesEvaluated) {
    examplesEvaluated = batchSize * totalBatchesEvaluated;
    document.getElementById("examplesEvaluated").innerHTML = `Examples evaluated: ${examplesEvaluated}`
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

        displayAccuracy(avgCost)
    }

}

// function displayCost(cost, which) {
//     if (which === 'disc') {
//         costChartData.push({
//             x: graphRunner.getTotalBatchesTrained(),
//             y: cost.get()
//         });
//         costChart.update();
//     } else {
//         accuracyChartData.push({
//             x: graphRunner.getTotalBatchesTrained(),
//             y: cost.get()
//         });
//         accuracyChart.update();
//     }
// }


var accuracyGraph = new cnnvis.Graph();
var accuracyWindow = new cnnutil.Window(100);

function displayAccuracy(accuracy) {

    var accuracy = accuracy.get() * 100;
    var batchesTrained = graphRunner.getTotalBatchesTrained();

    accuracyWindow.add(accuracy);

    var xa = accuracyWindow.get_average();

    if (xa >= 0) { // if they are -1 it means not enough data was accumulated yet for estimates
        accuracyGraph.add(batchesTrained, xa);
        accuracyGraph.drawSelf(document.getElementById("accuracygraph"));
    }
}

function displayInferenceExamplesPerSec(examplesPerSec) {
    inferencesPerSec =
        smoothExamplesPerSec(inferencesPerSec, examplesPerSec);
    inferenceDuration = Number((1000 / examplesPerSec).toPrecision(3));

    generationsPerSec = inferencesPerSec;
    generationDuration = inferenceDuration;
}


var examplesPerSecGraph = new cnnvis.Graph();
var examplesPerSecWindow = new cnnutil.Window(100);

function displayExamplesPerSec(_examplesPerSec) {


    var batchesTrained = graphRunner.getTotalBatchesTrained();

    examplesPerSecWindow.add(_examplesPerSec);

    var xa = examplesPerSecWindow.get_average();

    if (xa >= 0) { // if they are -1 it means not enough data was accumulated yet for estimates
        examplesPerSecGraph.add(batchesTrained, xa);
        examplesPerSecGraph.drawSelf(document.getElementById("examplespersecgraph"));
    }

    examplesPerSec =
        smoothExamplesPerSec(examplesPerSec, _examplesPerSec);

    document.getElementById("examplesPerSec").innerHTML = `Examples/sec: ${examplesPerSec}`;
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

    let realImages = [];
    const realLabels = [];
    // const realLogits = [];

    let fakeImages = [];
    const fakeLabels = [];
    // const fakeLogits = [];

    for (let i = 0; i < inputFeeds.length; i++) {
        realImages.push(inputFeeds[i][0].data);
        fakeImages.push((inferenceOutputs[0][i]));

    }

    realImages =
        dataSet.unnormalizeExamples(realImages, IMAGE_DATA_INDEX);

    fakeImages =
        dataSet.unnormalizeExamples(fakeImages, IMAGE_DATA_INDEX);

    // Draw the images.
    for (let i = 0; i < inputFeeds.length; i++) {
        inputNDArrayVisualizers[i].saveImageDataFromNDArray(realImages[i]);
        fakeInputNDArrayVisualizers[i].saveImageDataFromNDArray(fakeImages[i]);
    }

    // Draw the logits.
    for (let i = 0; i < inputFeeds.length; i++) {

        inputNDArrayVisualizers[i].draw();
        fakeInputNDArrayVisualizers[i].draw();

    }
}

function addLayer(which) {

    const modelLayer = new ModelLayer(); //document.createElement('model-layer');

    if (which === 'gen') {

        const lastHiddenLayer = genHiddenLayers[genHiddenLayers.length - 1];
        const lastOutputShape = lastHiddenLayer != null ?
            lastHiddenLayer.getOutputShape() :
            randVectorShape;
        genHiddenLayers.push(modelLayer);

        modelLayer.initialize(window, lastOutputShape, which);

        // genLayersContainer.appendChild(modelLayer.paramContainer);
    } else if (which === 'disc') {

    } else { // critic
        const lastHiddenLayer = critHiddenLayers[critHiddenLayers.length - 1];
        const lastOutputShape = lastHiddenLayer != null ?
            lastHiddenLayer.getOutputShape() :
            inputShape;
        critHiddenLayers.push(modelLayer);

        modelLayer.initialize(window, lastOutputShape, which);

        // critLayersContainer.appendChild(modelLayer.paramContainer);
    }



    return modelLayer;
}

function removeAllLayers(which) {
    if (which === 'gen') {
        for (let i = 0; i < genHiddenLayers.length; i++) {
            genLayersContainer.removeChild(genHiddenLayers[i].paramContainer);
        }
        genHiddenLayers = [];
    } else if (which === 'disc') {

    } else {
        for (let i = 0; i < critHiddenLayers.length; i++) {
            critLayersContainer.removeChild(critHiddenLayers[i].paramContainer);
        }
        critHiddenLayers = [];

    }

    layerParamChanged();
}

function validateModel() {
    let valid = true;

    for (let i = 0; i < genHiddenLayers.length; ++i) {
        valid = valid && genHiddenLayers[i].isValid();
    }
    if (genHiddenLayers.length > 0) {
        const lastLayer = genHiddenLayers[genHiddenLayers.length - 1];
        valid = valid &&
            util.arraysEqual(inputShape, lastLayer.getOutputShape());
    }
    valid = valid && (genHiddenLayers.length > 0);

    for (let i = 0; i < critHiddenLayers.length; ++i) {
        valid = valid && critHiddenLayers[i].isValid();
    }
    if (critHiddenLayers.length > 0) {
        const lastLayer = critHiddenLayers[critHiddenLayers.length - 1];
        valid = valid &&
            util.arraysEqual(labelShape, lastLayer.getOutputShape());
    }
    valid = valid && (critHiddenLayers.length > 0);

    isValid = valid;
}

function layerParamChanged() {
    // Go through each of the model layers and propagate shapes.
    let lastOutputShape = randVectorShape;
    for (let i = 0; i < genHiddenLayers.length; i++) {
        lastOutputShape = genHiddenLayers[i].setInputShape(lastOutputShape);
    }

    lastOutputShape = inputShape;
    for (let i = 0; i < critHiddenLayers.length; i++) {
        lastOutputShape = critHiddenLayers[i].setInputShape(lastOutputShape);
    }

    validateModel();

    if (isValid) {
        createModel();
    }
}

function uploadModel() {
    (document.querySelector('#model-file')).click();
}

function setupUploadModelButton() {
    // Show and setup the load view button.
    const fileInput = document.querySelector('#model-file');
    fileInput.addEventListener('change', event => {
        const file = fileInput.files[0];
        // Clear out the value of the file chooser. This ensures that if the user
        // selects the same file, we'll re-read it.
        fileInput.value = '';
        const fileReader = new FileReader();
        fileReader.onload = (evt) => {
            removeAllLayers('gen');
            const modelJson = fileReader.result;
            loadModelFromJson(modelJson, 'gen');
        };
        fileReader.readAsText(file);
    });
}

function getModelAsJson() {
    const layerBuilders = [];
    for (let i = 0; i < genHiddenLayers.length; i++) {
        layerBuilders.push(genHiddenLayers[i].layerBuilder);
    }
    return JSON.stringify(layerBuilders);
}

function loadModelFromJson(modelJson, which) {
    var lastOutputShape;
    var hiddenLayers;
    if (which === 'disc') {

    } else if (which === 'gen') {
        lastOutputShape = randVectorShape;
        hiddenLayers = genHiddenLayers;
    } else {
        lastOutputShape = inputShape;
        hiddenLayers = critHiddenLayers;
    }

    const layerBuilders = JSON.parse(modelJson);
    for (let i = 0; i < layerBuilders.length; i++) {
        const modelLayer = addLayer(which);
        modelLayer.loadParamsFromLayerBuilder(lastOutputShape, layerBuilders[i]);
        lastOutputShape = hiddenLayers[i].setInputShape(lastOutputShape);

    }
    validateModel();
}

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
            loadWeightsFromJson(weightsJson);
            createModel();
        };
        fileReader.readAsText(file);
    });
}

function loadWeightsFromJson(weightsJson) {
    genloadedWeights = JSON.parse(weightsJson);
}



function run() {

    critLearningRate = 0.01;
    genLearningRate = 0.01;
    critMomentum = 0.1;
    genMomentum = 0.1;
    critNeedMomentum = false;
    genNeedMomentum = false;
    critGamma = 0.1;
    genGamma = 0.1;
    critBeta1 = 0.9;
    critBeta2 = 0.999;
    genBeta1 = 0.9;
    genBeta2 = 0.999;
    critNeedGamma = false;
    genNeedGamma = false;
    critNeedBeta = false;
    genNeedBeta = true;
    batchSize = 15;

    updateNetParamDisplay();

    // var normalizationDropdown = document.getElementById("normalization-dropdown");
    // normalizationDropdown.options[selectedNormalizationOption].selected = 'selected';

    var envDropdown = document.getElementById("environment-dropdown");
    selectedEnvName = 'GPU';
    var ind = indexOfDropdownOptions(envDropdown.options, selectedEnvName)
    envDropdown.options[ind].selected = 'selected';
    updateSelectedEnvironment(selectedEnvName, graphRunner);


    // Default optimizer is momentum
    genSelectedOptimizerName = "adam";
    critSelectedOptimizerName = "adam";

    // var genOptimizerDropdown = document.getElementById("gen-optimizer-dropdown");
    // var ind = indexOfDropdownOptions(genOptimizerDropdown.options, genSelectedOptimizerName);
    // genOptimizerDropdown.options[ind].selected = 'selected';

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
    graphRunner = new MyGraphRunner(math, session, eventObserver);

    // Set up datasets.
    populateDatasets();

    // document.querySelector('#gen-optimizer-dropdown').addEventListener('change', (event) => {
    //     // Activate, deactivate hyper parameter inputs.
    //     refreshHyperParamRequirements(event.target.value, 'gen');
    //     genSelectedOptimizerName = event.target.value;
    //     console.log('gen optimizer =', event.target.value)
    // });

    applicationState = ApplicationState.IDLE;
    loadedWeights = null;
    genLoadedWeights = null;
    modelInitialized = false;
    showTrainStats = false;
    showDatasetStats = false;

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
    critHiddenLayers = [];
    genHiddenLayers = [];
    examplesPerSec = 0;
    evalExamplesPerSec = 0;
    inferencesPerSec = 0;
    generationsPerSec = 0;
    randVectorShape = [100];
}


function updateSelectedEnvironment(selectedEnvName, _graphRunner = null) {

    math = (selectedEnvName === 'GPU') ? mathGPU : mathCPU;
    console.log('math =', math === mathGPU ? 'mathGPU' : 'mathCPU')
    if (_graphRunner != null) {
        _graphRunner.setMath(math);
    }

}

var updateNetParamDisplay = function () {
    // document.getElementById('crit-learning-rate-input').value = critLearningRate;
    // document.getElementById('batch_size_input').value = batchSize;
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

    updateNetParamDisplay();
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
var btn_eval = document.getElementById('buttoneval');
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
            // btn_train.disabled = true;
            btn_train.style.visibility = 'hidden';
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

            run();

            monitor();

        }, 0);

    } else {
        console.log('device/webgl not supported')
        btn_infer.disabled = true;
        // btn_train.disabled = true;
        btn_eval.disabled = true;
    }
}