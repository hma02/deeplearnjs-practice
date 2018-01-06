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


var Array3D = dl.Array3D;
var ENV = dl.ENV;

var ApplicationState = {
    IDLE: 1,
    TRAINING: 2
}

const CONTENT_NAMES = ['stata', 'Upload from file'];
const STYLE_MAPPINGS = {
    'Udnie, Francis Picabia': 'udnie',
    'Rain Princess, Leonid Afremov': 'rain_princess',
    'The Wave, Katsushika Hokusai': 'wave'
};
const STYLE_NAMES = Object.keys(STYLE_MAPPINGS);

// Polymer properties
var contentNames;
var selectedContentName;
var styleNames;
var selectedStyleName;
var status;
var applicationState;

var transformNet;

// DOM Elements
var contentImgElement;
var styleImgElement;

var sizeSlider;

var canvas;

var startButton;

var camDialog;
var stream;
var webcamVideoElement;
var takePicButton;
var closeModal;

var fileSelect;

function initWebcamVariables() {
    camDialog = document.querySelector('#webcam-dialog');
    webcamVideoElement = document.querySelector('#webcamVideo');
    takePicButton = document.querySelector('#takePicButton');
    closeModal = document.querySelector('#closeModal');

    // Check if webcam is even available
    const navigatorAny = navigator;
    navigator.getUserMedia = navigator.getUserMedia ||
        navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
        navigatorAny.msGetUserMedia;
    if (navigator.getUserMedia) {
        contentNames.push('Use webcam');
    }

    closeModal.addEventListener('click', () => {
        stream.getTracks()[0].stop();
        $('#webcam-dialog').modal('hide');
    });

    takePicButton.addEventListener('click', () => {
        const hiddenCanvas =
            document.querySelector('#hiddenCanvas');
        const hiddenContext =
            hiddenCanvas.getContext('2d');
        hiddenCanvas.width = webcamVideoElement.width;
        hiddenCanvas.height = webcamVideoElement.height;
        hiddenContext.drawImage(
            webcamVideoElement, 0, 0, hiddenCanvas.width,
            hiddenCanvas.height);
        const imageDataURL = hiddenCanvas.toDataURL('image/jpg');
        contentImgElement.src = imageDataURL;
        stream.getTracks()[0].stop();
    });
}

function openWebcamModal() {
    $('#webcam-dialog').modal()
    navigator.getUserMedia({
            video: true
        },
        (_stream) => {
            stream = _stream;
            webcamVideoElement.src = window.URL.createObjectURL(stream);
            webcamVideoElement.play();
        },
        (err) => {
            console.error(err);
        });
}

function runInference() {
    ENV.math.scope(async() => {
        const preprocessed = Array3D.fromPixels(contentImgElement);
        const inferenceResult = await transformNet.predict(preprocessed);
        setCanvasShape(inferenceResult.shape);
        renderToCanvas(inferenceResult, canvas);
    });
}

function setCanvasShape(shape) {
    canvas.width = shape[1];
    canvas.height = shape[0];
    if (shape[1] > shape[0]) {
        canvas.style.width = '500px';
        canvas.style.height = (shape[0] / shape[1] * 500).toString() + 'px';
    } else {
        canvas.style.height = '500px';
        canvas.style.width = (shape[1] / shape[0] * 500).toString() + 'px';
    }
}


function renderToCanvas(a, canvas) {
    const [height, width, ] = a.shape;
    const ctx = canvas.getContext('2d');
    const imageData = new ImageData(width, height);
    const data = a.dataSync();
    for (let i = 0; i < height * width; ++i) {
        const j = i * 4;
        const k = i * 3;
        imageData.data[j + 0] = Math.round(255 * data[k + 0]);
        imageData.data[j + 1] = Math.round(255 * data[k + 1]);
        imageData.data[j + 2] = Math.round(255 * data[k + 2]);
        imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
}


function run() {

    // Initialize deeplearn.js stuff
    canvas = document.querySelector('#imageCanvas');

    // Initialize polymer properties
    applicationState = ApplicationState.IDLE;
    status = '';

    // Retrieve DOM for images
    contentImgElement =
        document.querySelector('#contentImg');
    styleImgElement = document.querySelector('#styleImg');

    // Render DOM for images
    contentNames = CONTENT_NAMES.slice();
    selectedContentName = 'stata';
    contentImgElement.src = 'deeplearnjs/style_transfer/images/stata.jpg';
    contentImgElement.height = 250;

    styleNames = STYLE_NAMES;
    selectedStyleName = 'Udnie, Francis Picabia';
    styleImgElement.src = 'deeplearnjs/style_transfer/images/udnie.jpg';
    styleImgElement.height = 250;

    transformNet =
        new TransformNet(STYLE_MAPPINGS[selectedStyleName]);

    initWebcamVariables();

    // tslint:disable-next-line:no-any
    // sizeSlider = document.querySelector('#sizeSlider');
    // sizeSlider.addEventListener(
    //     'immediate-value-change',
    //     // tslint:disable-next-line:no-any
    //     (event) => {
    //         styleImgElement.height = sizeSlider.immediateValue;
    //         contentImgElement.height = sizeSlider.immediateValue;
    //     });
    // // tslint:disable-next-line:no-any
    // sizeSlider.addEventListener('change', (event) => {
    //     styleImgElement.height = sizeSlider.immediateValue;
    //     contentImgElement.height = sizeSlider.immediateValue;
    // });

    fileSelect = document.querySelector('#fileSelect');
    // tslint:disable-next-line:no-any
    fileSelect.addEventListener('change', (event) => {
        const f = event.target.files[0];
        const fileReader = new FileReader();
        fileReader.onload = ((e) => {
            const target = e.target;
            contentImgElement.src = target.result;
        });
        fileReader.readAsDataURL(f);
        fileSelect.value = '';
    });

    // Add listener to drop downs
    const contentDropdown = document.querySelector('#content-dropdown');

    for (var i = 0; i < contentNames.length; i++) {
        var o = document.createElement("option");
        o.value = contentNames[i];
        o.text = contentNames[i];
        contentDropdown.appendChild(o);
    }

    // tslint:disable-next-line:no-any
    contentDropdown.addEventListener('change', (event) => {
        const selected = event.target.value
        if (selected === 'Use webcam') {
            openWebcamModal();
        } else if (selected === 'Upload from file') {
            fileSelect.click();
        } else {
            contentImgElement.src = 'deeplearnjs/style_transfer/images/' + selected + '.jpg';
        }
    });

    const styleDropdown = document.querySelector('#style-dropdown');

    for (var i = 0; i < styleNames.length; i++) {
        var o = document.createElement("option");
        o.value = styleNames[i];
        o.text = styleNames[i];
        styleDropdown.appendChild(o);
    }

    styleDropdown.addEventListener('change', (event) => {
        styleImgElement.src =
            'deeplearnjs/style_transfer/images/' + STYLE_MAPPINGS[event.target.value] + '.jpg';
    });

    // Add listener to start
    startButton = document.querySelector('#btnstart');
    startButton.addEventListener('click', () => {
        (document.querySelector('#load-error-message')).style.display =
            'none';
        startButton.textContent =
            'Starting style transfer.. Downloading + running model';
        startButton.disabled = true;
        transformNet.setStyle(STYLE_MAPPINGS[selectedStyleName]);

        transformNet.load()
            .then(() => {
                startButton.textContent = 'Processing image';
                runInference();
                startButton.textContent = 'Start Style Transfer';
                startButton.disabled = false;
            })
            .catch((error) => {
                console.log(error);
                startButton.textContent = 'Start Style Transfer';
                startButton.disabled = false;
                const errMessage =
                    document.querySelector('#load-error-message');
                errMessage.textContent = error;
                errMessage.style.display = 'block';
            });
        ga('send', 'event', 'deeplearn_style_transfer', 'click', 'Start Style Transfer', 0);
    });
}

function monitor() {

}

function start() {

    supported = detect_support();
    supported = true;

    btn = document.getElementById("btnstart");

    if (supported) {
        console.log('device & webgl supported')
        btn.disabled = false;

        run();
        monitor();

    } else {
        console.log('device/webgl not supported')
        btn.disabled = true;
    }

}