function startEvalulating() {
    const data = getImageDataOnly();

    // Recreate optimizer with the selected optimizer and hyperparameters.
    critOptimizer = createOptimizer('crit'); // for js, exact same optimizer
    // genOptimizer = createOptimizer('gen');

    if (isValid && data != null) {
        // recreateCharts();
        graphRunner.resetStatistics();

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

        const critFeeds = [{
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

        const genFeeds = [{
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

        graphRunner.evaluate(
            critLoss, null, critFeeds, genFeeds, batchSize,
            critOptimizer, null, undefined, COST_INTERVAL_MS);

        showEvalStats = true;
        applicationState = ApplicationState.Evaluating;
    }
}


// function createCritModel() {
//     if (session != null) {
//         session.dispose();
//     }

//     critModelInitialized = false;
//     if (isValid === false) {
//         return;
//     }

//     // Construct graph
//     graph = new Graph();
//     const g = graph;
//     randomTensor = g.placeholder('random', randVectorShape);
//     xTensor = g.placeholder('input', inputShape);
//     oneTensor = g.placeholder('one', [2]);
//     zeroTensor = g.placeholder('zero', [2]);

//     const varianceInitializer = new VarianceScalingInitializer()
//     const zerosInitializer = new ZerosInitializer()
//     const onesInitializer = new OnesInitializer();

//     // Construct generator
//     let gen = randomTensor;
//     for (let i = 0; i < genHiddenLayers.length; i++) {
//         let weights = null;
//         if (genLoadedWeights != null) {
//             weights = genLoadedWeights[i];
//         } // can reload gen weights if needed, typically after gen is trained, then download to json and load back after clicking
//         [gen] = genHiddenLayers[i].addLayerMultiple(g, [gen],
//             'generator', weights);
//     }
//     gen = g.tanh(gen); // fake image

//     // Construct critic
//     let crit1 = gen;
//     let crit2 = xTensor; // real image
//     for (let i = 0; i < critHiddenLayers.length; i++) {
//         let weights = null;
//         // if (loadedWeights != null) {
//         //     weights = loadedWeights[i];
//         // } // always need to retrain critic (which is the process of eval), never load weights for critic
//         [crit1, crit2] = critHiddenLayers[i].addLayerMultiple(g, [crit1, crit2],
//             'critic', weights);
//     }

//     critPredictionReal = crit2;
//     critPredictionFake = crit1;
//     generatedImage = gen;
//     const critLossReal = g.softmaxCrossEntropyCost(
//         critPredictionReal,
//         oneTensor
//     );
//     const critLossFake = g.softmaxCrossEntropyCost(
//         critPredictionFake,
//         zeroTensor
//     );
//     critLoss = g.add(critLossReal, critLossFake);

//     genLoss = g.softmaxCrossEntropyCost(
//         critPredictionFake,
//         oneTensor
//     );

//     session = new Session(g, math);
//     graphRunner.setSession(session);

//     // startInference();

//     critModelInitialized = true;
// }