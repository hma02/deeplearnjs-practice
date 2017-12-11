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

var dl = deeplearn;
var NDArrayMathGPU = dl.NDArrayMathGPU;
var Array3D = dl.Array3D;
var Array4D = dl.Array4D;
var Array1D = dl.Array1D;
var ENV = dl.ENV;

function computeWeightsShape4D(
    inputDepth, outputDepth, filterHeight,
    filterWidth) {
    return [filterHeight, filterWidth, inputDepth, outputDepth];
}

class ConvBenchmark {

    constructor(libName, params, sizeRange) {
        this.libName = libName
        this.index = this.libName === 'dljs' ? 0 : 1;
        this.params = params;
        this.sizeRange = sizeRange

        this.size = sizeRange;
        this.sizes = [];

        this.paused = true;
        this.request = false;

        this.btn = document.getElementById('buttontp' + "_" + this.libName);
        this.btn.addEventListener('click', () => {
            this.paused = !this.paused;

            if (this.paused) {
                // this.stop_test(); // can return quickly
            } else {
                this.request = true;
            }

            ga('send', 'event', 'deeplearn_conv_benchmark', 'click', `Run Benchmark ${this.libName}`, this.libName === 'dljs' ? 30 : 31);
        });
    }

    changeSizeRange(sizeRange) {
        this.sizeRange = sizeRange;
        this.size = this.sizeRange;
        this.sizes = [];
        chartData[this.index].data = [];
        config.data.datasets = chartData;
        chart.update();
    }

    monitorRequestAndUpdateUI() {

        if (this.paused) {
            if (this.sizes.length > 0) {
                this.btn.value = 'Resume';
            } else {
                this.btn.value = 'Run Benchmark';
            }
        } else {
            this.btn.value = 'Pause'
        }

        if (this.request) {
            this.request = false;

            if (this.sizes.length > 0) {
                this.run_test(); //this.resume_test();
            } else {

                if (chartData[this.index].data.length > 0) {
                    chartData[this.index].data = [];
                }
                this.run_test();
            }

        }
    }

    run(size, opType, params) {

        if (this.libName === 'dljs') {
            var math = new NDArrayMathGPU();
            var gpgpu = math.getGPGPUContext();
        }

        const inDepth = params.inDepth;
        const inShape = [size, size, inDepth];
        const outDepth = params.outDepth;
        const filterSize = params.filterSize;
        const stride = params.stride;
        let zeroPad = params.zeroPad;

        // outputRows>=0 needs to be asserted
        var outputRows = (inShape[0] - filterSize + 2 * zeroPad) / stride + 1;

        function isInt(a) {
            return a % 1 === 0;
        }

        if (outputRows <= 0) {
            // console.log(`input size ${size} doesn't satisfy assertion outputRows>0, given inputRows=${size}, filterSize=${filterSize}, zeroPad=${zeroPad}, stride=${stride}, minimal size needs to be ${filterSize + 2 * zeroPad - stride}`)
        } else if (!isInt(outputRows)) {
            // console.log(`outputRows ${outputRows} is not int`)
        }

        function find_min_pad(i, f, s) {

            for (let n = 0;; n++) {
                var z = (n * s - i + f) / 2;
                if (z > 0) {
                    // console.log('found suitable z=' + z)
                    return z
                }
            }

        }
        zeroPad = find_min_pad(size, filterSize, stride);
        // console.log(`min pad ${zeroPad} applied`)

        let benchmark;
        let out;
        let b;
        let x;

        if (this.libName === 'dljs') { //deeplearnjs

            if (opType === 'regular') {
                x = Array3D.randUniform(inShape, -1, 1);
                const wShape = computeWeightsShape4D(
                    inDepth, outDepth, filterSize, filterSize);
                var W = Array4D.randUniform(wShape, -1, 1);
                // const b = Array1D.randUniform([outDepth], -1, 1);
                // console.log(W.getValues(), W.shape, x.getValues(), x.shape)

                benchmark = () => math.conv2d(x, W, null, stride, zeroPad); //bias=null, pad =0, this padding will be applied on four borders of input : left right top bottom

            } else if (opType === 'transposed') {

                x = Array3D.randUniform([size, size, regParams.outDepth], -1, 1);
                const wShape = computeWeightsShape4D(
                    inDepth, params.outDepth, filterSize, filterSize);
                W = Array4D.randUniform(wShape, -1, 1);

                // no bias for conv transposed
                benchmark = () => math.conv2dTranspose(x, W, [size, size, inDepth], stride, pad);

            } else if (opType === 'depthwise') {
                x = Array3D.randUniform(inShape, -1, 1);
                const wShape = computeWeightsShape4D(
                    inDepth, params.channelMul, filterSize, filterSize);
                W = Array4D.randUniform(wShape, -1, 1);

                // no bias for depth wise conv
                benchmark = () => math.depthwiseConv2D(x, W, stride, pad);
            } else {
                throw new Error(`Unknown option ${opType}`);
            }

        } else { // convnetjs

            console.assert(opType === 'regular', `unsupported conv op type`)

            x = new convnetjs.Vol(size, size, inDepth); // 128,128,3 
            const opt = {
                in_sx: size,
                in_sy: size,
                in_depth: inDepth,
                sx: filterSize,
                filters: outDepth,
                stride: stride,
                pad: zeroPad // this will be applied on four borders of input :left right top bottom
            }; // no bias, pad=0
            var layer = new convnetjs.ConvLayer(opt);

            // console.log(layer)

            benchmark = () => layer.forward(x);
        }

        let totalTime;

        if (this.libName === 'dljs') {

            // Warmup.
            if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED')) {
                gpgpu.runQuery(benchmark);
            } else {
                out = benchmark();
                out.dataSync();
                out.dispose();
            }

            if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE')) {
                totalTime = gpgpu.runQuery(benchmark);
            } else {
                const start = performance.now();

                out = benchmark();
                out.dataSync();

                totalTime = performance.now() - start;
                out.dispose();
            }

        } else {

            const start = performance.now();

            out = benchmark();
            totalTime = performance.now() - start;
            out = null;
        }

        // console.log(`${this.libName} convolution output: ${out}`)

        const cleanup = () => {

            if (this.libName === 'dljs') {
                x.dispose();
                W.dispose();
                if (b != null) {
                    b.dispose();
                }
            } else {
                x = null;
            }

        };

        cleanup();
        return totalTime;
    }

    displayResult(size, _time) {

        console.log(size, _time);

        let time = _time.toFixed(3);

        chartData[this.index].data.push({
            x: size,
            y: time
        });

        // sort the array again, in case sizes do not come in ascending order from the async promises
        chartData[this.index].data.sort(function (a, b) {
            // Compare the 2 dates
            if (a.x < b.x) return -1;
            if (a.x > b.x) return 1;
            return 0;
        });

        config.data.datasets = chartData;

        chart.update();

        var times = [];
        var sizes = [];
        chartData[this.index].data.forEach(dic => {
            times.push(dic.y);
            sizes.push(dic.x)
        })
        update_table_col(table, this.index + 1, times); // col0 is the size labels
        update_table_col(table, 0, sizes);

    }
    run_test() {

        if (this.paused == true) {
            return
        }

        let current_size = this.size

        if (current_size > 128 * this.sizeRange) {

            this.size = this.sizeRange;
            this.sizes = [];

            this.btn.click(); //toggle_pause();
            return
        }

        this.sizes.push(current_size)

        let t = this.run(current_size, 'regular', this.params)

        if (typeof (t.then) === "undefined") {
            this.displayResult(current_size, t);
        } else if (t.then instanceof Function) {
            t.then(_t => this.displayResult(current_size, _t));
        } else {
            throw new Error('unrecognized t');
        }

        this.size = this.size * 2;
        requestAnimationFrame(() => this.run_test());
    }

}

var config;
var chart;
var table;

var sizeRange;
var convParams;
var bms = [];

function run() {

    const canvas = document.getElementById(`plot`);

    canvas.width = 400;
    canvas.height = 300;
    const context = canvas.getContext('2d');

    config = {
        type: 'line',
        data: {
            datasets: [{
                data: [],
                fill: false,
                label: ' ',
                pointRadius: 0,
                borderColor: 'rgba(75,192,192,1)',
                backgroundColor: 'rgba(75,192,192,1)',
                borderWidth: 1,
                // lineTension: 0,
                // pointHitRadius: 8
            }]
        },
        options: {
            animation: {
                duration: 0
            },
            responsive: true,
            scales: {
                xAxes: [{
                    type: 'linear',
                    position: 'bottom'
                }],
                yAxes: [{
                    ticks: {
                        min: null,
                        callback: (label, index, labels) => {
                            let num = Number(label).toFixed(2);
                            return `${num}`;
                        }
                    }
                }]
            }
        }
    };

    chart = new Chart(context, config);

    chartData = [{
            label: 'deeplearnjs -- t1',
            backgroundColor: window.chartColors.red,
            borderColor: window.chartColors.red,
            data: [],
            fill: false,
            pointRadius: 3,
            pointHitRadius: 5,
            borderWidth: 1,
            // lineTension: 0,
        },
        {
            label: "convnetjs -- t2",
            backgroundColor: window.chartColors.blue,
            borderColor: window.chartColors.blue,
            data: [],
            fill: false,
            pointRadius: 3,
            pointHitRadius: 5,
            borderWidth: 1,
            // lineTension: 0,
        }
    ]
    config.data.datasets = chartData;
    chart.update();

    table = document.getElementById(`divTable`);
    init_table(table, ['Size', 't1 (ms)', 't2 (ms)']);


    var sizeRangeDropdown = document.getElementById("range-dropdown");
    sizeRange = 16;
    var ind = indexOfDropdownOptions(sizeRangeDropdown.options, sizeRange)
    sizeRangeDropdown.options[ind].selected = 'selected';

    sizeRangeDropdown.addEventListener('change', (event) => {
        sizeRange = event.target.value;
        bms.forEach(bm => bm.changeSizeRange(Number(sizeRange)))
        console.log(`change size range to [${sizeRange}, ${sizeRange * 128}]`)
    });


    // (inputRows - fieldSize + 2 * zeroPad) / stride + 1) >=0 needs to be asserted
    const convParams = {
        inDepth: 8,
        outDepth: 3,
        filterSize: 7,
        stride: 2,
        zeroPad: 0 // adjust zeroPad so that (inputSize - filterSize + 2* zeroPad) is integer
    };

    bms.push(new ConvBenchmark('dljs', convParams, sizeRange));
    bms.push(new ConvBenchmark('cnjs', convParams, sizeRange));

}

function monitor() {

    bms.forEach(bm => bm.monitorRequestAndUpdateUI());

    setTimeout(function () {
        monitor();
    }, 100);
}

function start() {

    supported = detect_support();
    supported = true;

    if (supported) {
        console.log('device & webgl supported')
        document.getElementById("buttontp_dljs").disabled = false;
    } else {
        console.log('device/webgl not supported')
        document.getElementById("buttontp_dljs").disabled = true;
    }

    run();
    monitor();
}