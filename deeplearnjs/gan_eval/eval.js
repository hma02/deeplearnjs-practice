function startTraining() {
    const data = getImageDataOnly();

    // Recreate optimizer with the selected optimizer and hyperparameters.
    discOptimizer = createOptimizer('disc');
    genOptimizer = createOptimizer('gen');

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

        const discFeeds = [{
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

        graphRunner.train(
            discLoss, genLoss, discFeeds, genFeeds, batchSize,
            discOptimizer, genOptimizer, undefined, COST_INTERVAL_MS);

        showTrainStats = true;
        applicationState = ApplicationState.TRAINING;
    }
}