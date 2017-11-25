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

        // showEvalStats = true;
        // applicationState = ApplicationState.Evaluating;
    }
}