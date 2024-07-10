import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;
const OPENCV_URI = "https://docs.opencv.org/master/opencv.js";
const enableWebcamButton = document.getElementById("webcamButton");

const foreheadDisplay = document.getElementById("forehead-bpm");
const chinDisplay = document.getElementById("chin-bpm");
const mouthDisplay = document.getElementById("mouth-bpm");
const rightCheekDisplay = document.getElementById("right-cheek-bpm");
const leftCheekDisplay = document.getElementById("left-cheek-bpm");
const noseDisplay = document.getElementById("nose-bpm");
const nasionDisplay = document.getElementById("nasion-bpm");
const avirageDisplay = document.getElementById("average-bpm");


let faceLandmarker;
let runningMode = "IMAGE";
let webcamRunning = false;
const videoWidth = 480;

const timestamps =[];
const bufferSize = 180; // Store last 150 frames (5 seconds at 30 FPS)
const MAX_BUFFER_SIZE = 181;
const DEFAULT_FPS = 30;
const LOW_BPM = 42;
const HIGH_BPM = 240;
const SEC_PER_MIN = 60;
let counter = 0
let lastVideoTime = -1;
let results = undefined;

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

const FOREHEAD_LANDMARKS = [
  { "start": 9, "end": 107 },
  { "start": 107, "end": 69 },
  { "start": 69, "end": 104 },
  { "start": 104, "end": 103 },
  { "start": 103, "end": 67 },
  { "start": 67, "end": 109 },
  { "start": 109, "end": 10 },
  { "start": 10, "end": 338 },
  { "start": 338, "end": 297 },
  { "start": 297, "end": 332 },
  { "start": 332, "end": 333 },
  { "start": 333, "end": 299 },
  { "start": 299, "end": 336 },
  { "start": 336, "end": 9 }
];

const CHIN_LANDMARKS = [
  { "start": 152, "end": 148 },
  { "start": 148, "end": 176 },
  { "start": 176, "end": 140 },
  { "start": 140, "end": 32 },
  { "start": 32, "end": 194 },
  { "start": 194, "end": 182 },
  { "start": 182, "end": 83 },
  { "start": 83, "end": 18 },
  { "start": 18, "end": 313 },
  { "start": 313, "end": 406 },
  { "start": 406, "end": 418 },
  { "start": 418, "end": 262 },
  { "start": 262, "end": 369 },
  { "start": 369, "end": 400 },
  { "start": 400, "end": 377 },
  { "start": 377, "end": 152 }
];

const MOUTH_LANDMARKS = [
  { "start": 18, "end": 83 },
  { "start": 83, "end": 182 },
  { "start": 182, "end": 106 },
  { "start": 106, "end": 43 },
  { "start": 43, "end": 57 },
  { "start": 57, "end": 186 },
  { "start": 186, "end": 92 },
  { "start": 92, "end": 165 },
  { "start": 165, "end": 167 },
  { "start": 167, "end": 164 },
  { "start": 164, "end": 393 },
  { "start": 393, "end": 391 },
  { "start": 391, "end": 322 },
  { "start": 322, "end": 410 },
  { "start": 410, "end": 287 },
  { "start": 287, "end": 273 },
  { "start": 273, "end": 335 },
  { "start": 335, "end": 406 },
  { "start": 406, "end": 313 },
  { "start": 313, "end": 18 }
];

const RIGHT_CHEEK_LANDMARKS = [
  { "start": 149, "end": 150 },
  { "start": 150, "end": 136 },
  { "start": 136, "end": 172 },
  { "start": 172, "end": 58 },
  { "start": 58, "end": 132 },
  { "start": 132, "end": 93 },
  { "start": 93, "end": 234 },
  { "start": 234, "end": 227 },
  { "start": 227, "end": 116 },
  { "start": 116, "end": 111 },
  { "start": 111, "end": 117 },
  { "start": 117, "end": 118 },
  { "start": 118, "end": 101 },
  { "start": 101, "end": 205 },
  { "start": 205, "end": 207 },
  { "start": 207, "end": 212 },
  { "start": 212, "end": 202 },
  { "start": 202, "end": 204 },
  { "start": 204, "end": 211 },
  { "start": 211, "end": 170 },
  { "start": 170, "end": 149 }
];

const LEFT_CHEEK_LANDMARKS = [
  { "start": 378, "end": 395 },
  { "start": 395, "end": 431 },
  { "start": 431, "end": 424 },
  { "start": 424, "end": 422 },
  { "start": 422, "end": 432 },
  { "start": 432, "end": 427 },
  { "start": 427, "end": 425 },
  { "start": 425, "end": 330 },
  { "start": 330, "end": 347 },
  { "start": 347, "end": 346 },
  { "start": 346, "end": 340 },
  { "start": 340, "end": 345 },
  { "start": 345, "end": 447 },
  { "start": 447, "end": 454 },
  { "start": 454, "end": 323 },
  { "start": 323, "end": 361 },
  { "start": 361, "end": 288 },
  { "start": 288, "end": 397 },
  { "start": 397, "end": 365 },
  { "start": 365, "end": 379 },
  { "start": 379, "end": 378 }
];

const NOSE_LANDMARKS = [
  { "start": 1, "end": 44 },
  { "start": 44, "end": 237 },
  { "start": 237, "end": 218 },
  { "start": 218, "end": 219 },
  { "start": 219, "end": 235 },
  { "start": 235, "end": 64 },
  { "start": 64, "end": 49 },
  { "start": 49, "end": 209 },
  { "start": 209, "end": 126 },
  { "start": 126, "end": 217 },
  { "start": 217, "end": 174 },
  { "start": 174, "end": 196 },
  { "start": 196, "end": 197 },
  { "start": 197, "end": 419 },
  { "start": 419, "end": 399 },
  { "start": 399, "end": 437 },
  { "start": 437, "end": 355 },
  { "start": 355, "end": 429 },
  { "start": 429, "end": 279 },
  { "start": 279, "end": 278 },
  { "start": 278, "end": 439 },
  { "start": 439, "end": 438 },
  { "start": 438, "end": 457 },
  { "start": 457, "end": 274 },
  { "start": 274, "end": 1 }
];

const NASION_LANDMARKS = [
  { "start": 197, "end": 196},
  { "start": 196, "end": 122 },
  { "start": 122, "end": 193 },
  { "start": 193, "end": 55 },
  { "start": 55, "end": 107 },
  { "start": 107, "end": 9 },
  { "start": 9, "end": 336 },
  { "start": 336, "end": 285 },
  { "start": 285, "end": 417 },
  { "start": 417, "end": 351 },
  { "start": 351, "end": 419 },
  { "start": 419, "end": 197 }
];



const FACE_ZONES = [
  {label: "FOREHEAD", landmarks: FOREHEAD_LANDMARKS, display: foreheadDisplay}, 
  {label: "LEFT_CHEEK", landmarks: LEFT_CHEEK_LANDMARKS, display: leftCheekDisplay}, 
  {label: "RIGHT_CHEEK", landmarks: RIGHT_CHEEK_LANDMARKS, display:  rightCheekDisplay}, 
  {label: "NOSE", landmarks: NOSE_LANDMARKS, display: noseDisplay}, 
  {label: "NASION", landmarks: NASION_LANDMARKS, display: nasionDisplay}, 
  {label: "MOUTH", landmarks: MOUTH_LANDMARKS, display: mouthDisplay}, 
  {label: "CHIN", landmarks: CHIN_LANDMARKS, display: chinDisplay}
];

const frameBuffer = Array(FACE_ZONES.length).fill([]);
// const heartrateBuffer = Array(FACE_ZONES.length).fill([]);
const frameHeartrateBuffer = Array(FACE_ZONES.length).fill([]);
const heartrateBuffer = [];
const greenValues = Array(FACE_ZONES.length).fill([]);

async function loadOpenCv(uri) {
  return new Promise(function(resolve, reject) {
    console.log("starting to load opencv");
    var tag = document.createElement('script');
    tag.src = uri;
    tag.async = true;
    tag.type = 'text/javascript'
    tag.onload = () => {
      cv['onRuntimeInitialized'] = () => {
        console.log("opencv ready");
        resolve();
      }
    };
    tag.onerror = () => {
      throw new URIError("opencv didn't load correctly.");
    };
    var firstScriptTag = document.getElementsByTagName('script')[0];
    firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
  });
}
loadOpenCv(OPENCV_URI);
// Before we can use HandLandmarker class we must wait for it to finish loading.
async function createFaceLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
      delegate: "CPU"
    },
    outputFaceBlendshapes: true,
    runningMode,
    numFaces: 1
  });
}
createFaceLandmarker();
// Check if webcam access is supported.
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

// If webcam supported, add event listener to button for when user wants to activate it.
if (hasGetUserMedia()) {
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}
let enableButtonInterval = setInterval(() => {
  console.log("sadasas")
  if(faceLandmarker && cv) {
    enableWebcamButton.disabled = false;
  }
}, 1000)

// Enable the live webcam view and start detection.
function enableCam(event) {
  if (!faceLandmarker) {
    console.log("Wait! faceLandmarker not loaded yet.");
    return;
  }

  if (webcamRunning === true) {
    webcamRunning = false;
    enableWebcamButton.innerText = "ENABLE PREDICTIONS";
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = "DISABLE PREDICTIONS";
  }

  // getUsermedia parameters.
  const constraints = {
    video: true
  };

  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

const drawingUtils = new DrawingUtils(canvasCtx);

async function predictWebcam() {
  if (enableWebcamButton.disabled === false) {
    clearInterval(enableButtonInterval)
  }
  const radio = video.videoHeight / video.videoWidth;
  video.style.width = videoWidth + "px";
  video.style.height = videoWidth * radio + "px";
  canvasElement.style.width = videoWidth + "px";
  canvasElement.style.height = videoWidth * radio + "px";
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;
  
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await faceLandmarker.setOptions({ runningMode: runningMode });
  }

  let startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    results = await faceLandmarker.detectForVideo(video, startTimeMs);
  }
  if (results && results.faceLandmarks) {
    processFrame()
  }
  // Call this function again to keep predicting when the browser is ready.
  if (webcamRunning === true) {
    setInterval(window.requestAnimationFrame(predictWebcam), 250); 
  }
}


function processFrame ()  {
  // console.log("counter",counter++)
  try {
    let time = Date.now()
    
    FACE_ZONES.forEach((zone, index) => {
      let frameCanvas = document.createElement('canvas');
      frameCanvas.width = video.videoWidth;
      frameCanvas.height = video.videoHeight;
      let frameCtx = frameCanvas.getContext('2d');
      let points = [];
      frameCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
      
      for (const landmarks of results.faceLandmarks) {
        drawingUtils.drawConnectors(
          landmarks,
          zone.landmarks,
          { fill: "#C0C0C070", lineWidth: 1 }
        );
        const startPoint = landmarks[zone.landmarks[0].start];
        points.push(new cv.Point(startPoint.x * frameCanvas.width, startPoint.y * frameCanvas.height));
        
        for (const connection of zone.landmarks) {
          const point = landmarks[connection.end];
          points.push(new cv.Point(point.x * frameCanvas.width, point.y * frameCanvas.height));
          
        }
      }
      
      let frame = cv.imread(frameCanvas);
      let frameRGBA = new cv.Mat();
      cv.cvtColor(frame, frameRGBA, cv.COLOR_RGB2RGBA);
      frame.delete();
      
      // Создание маски для выделенной области
      let mask = new cv.Mat.zeros(frameRGBA.rows, frameRGBA.cols, cv.CV_8UC1);
      let pointsData = new cv.MatVector();
      let mv = new cv.Mat(points.length, 1, cv.CV_32SC2);
      points.flat().forEach(({ x, y }, idx) => {
        mv.intPtr(idx, 0)[0] = x;
        mv.intPtr(idx, 0)[1] = y;
      });
      
      pointsData.push_back(mv);
      mv.delete();
      
      // cv.polylines(mask, pointsData, false, new cv.Scalar(255,255,255), 1);
      cv.fillPoly(mask, pointsData, new cv.Scalar(255, 255, 255));
      pointsData.delete();
      
      let maskedFrame = new cv.Mat();
      cv.bitwise_and(frameRGBA, frameRGBA, maskedFrame, mask);
      frameRGBA.delete();
      
      // cv.imshow(`${FACE_ZONES[index].display.id}-canvas`, maskedFrame);
      // cv.imshow(`${FACE_ZONES[index].display.id}-mask-canvas`, mask);
      
      let means = cv.mean(maskedFrame, mask);
      mask.delete();
      maskedFrame.delete();
      
      frameBuffer[index].push(means.slice(0, 3));
      timestamps.push(time)
      
      while (frameBuffer[index].length > MAX_BUFFER_SIZE) {
        frameBuffer[index].shift();
        timestamps.shift();
      }
      
      if (frameBuffer[index].length === MAX_BUFFER_SIZE) {
        let bpm = computeHeartRate(frameBuffer[index]);
        frameHeartrateBuffer[index].push(parseInt(bpm));
        while (frameHeartrateBuffer[index].length > 100) {
          frameHeartrateBuffer[index].shift();
        }
        console.log("frameHeartrateBuffer[index]",frameHeartrateBuffer[index])
        const getAverageHeartrate = (numbers) => numbers.reduce((acc, number) => acc + number, 0) / numbers.length;
        FACE_ZONES[index].display.innerText = `${FACE_ZONES[index].display.id}: ${getAverageHeartrate(frameHeartrateBuffer[index]).toFixed(0)} BPM`;
        calculateAverage();
      }
      
      
      frameCanvas.remove(); 
    }) 
  } catch (error) {
    console.error("Error in processFrame: ", error);
  }
  
}

function computeHeartRate(frames) {
  // if (frames.length < bufferSize) {
  //   return;
  // }
  let fps = getFps(timestamps);
  let signal = cv.matFromArray(frames.length, 1, cv.CV_32FC3, [].concat.apply([], frames));
  
  denoise(signal);
  standardize(signal);
  detrend(signal, fps);
  movingAverage(signal, 3, Math.max(Math.floor(fps/6), 2));
  signal = selectGreen(signal);
  // drawTime(signal);
  timeToFrequency(signal, true);
  // Calculate band spectrum limits
  let low = Math.floor(signal.rows * LOW_BPM / SEC_PER_MIN / fps);
  let high = Math.ceil(signal.rows * HIGH_BPM / SEC_PER_MIN / fps);
  if (!signal.empty()) {
    // Mask for infeasible frequencies
    let bandMask = cv.matFromArray(signal.rows, 1, cv.CV_8U,
      new Array(signal.rows).fill(0).fill(1, low, high+1));
      
    // drawFrequency(signal, low, high, bandMask);
    let result = cv.minMaxLoc(signal, bandMask);
    bandMask.delete();
    // console.log("result.maxLoc.y",result.maxLoc.y, "fps",fps, "signal.rows", signal.rows, "SEC_PER_MIN",SEC_PER_MIN)
    
    return (result.maxLoc.y * fps / signal.rows * SEC_PER_MIN).toFixed(0);
    // return result.maxLoc.y * fps / signal.rows * SEC_PER_MIN;
    
    // Draw BPM
    // drawBPM(bpm);
  }
  signal.delete();

}


function denoise(signal, rescan) {
  let diff = new cv.Mat();
  cv.subtract(signal.rowRange(1, signal.rows), signal.rowRange(0, signal.rows-1), diff);
  for (var i = 1; i < signal.rows; i++) {
    // if (rescan[i] == true) {
      let adjV = new cv.MatVector();
      let adjR = cv.matFromArray(signal.rows, 1, cv.CV_32FC1,
        new Array(signal.rows).fill(0).fill(diff.data32F[(i-1)*3], i, signal.rows));
      let adjG = cv.matFromArray(signal.rows, 1, cv.CV_32FC1,
        new Array(signal.rows).fill(0).fill(diff.data32F[(i-1)*3+1], i, signal.rows));
      let adjB = cv.matFromArray(signal.rows, 1, cv.CV_32FC1,
        new Array(signal.rows).fill(0).fill(diff.data32F[(i-1)*3+2], i, signal.rows));
      adjV.push_back(adjR); adjV.push_back(adjG); adjV.push_back(adjB);
      let adj = new cv.Mat();
      cv.merge(adjV, adj);
      cv.subtract(signal, adj, signal);
      adjV.delete(); adjR.delete(); adjG.delete(); adjB.delete();
      adj.delete();
    // }
  }
  diff.delete();
}
// Standardize signal
function standardize(signal) {
  let mean = new cv.Mat();
  let stdDev = new cv.Mat();
  let t1 = new cv.Mat();
  cv.meanStdDev(signal, mean, stdDev, t1);
  let means_c3 = cv.matFromArray(1, 1, cv.CV_32FC3, [mean.data64F[0], mean.data64F[1], mean.data64F[2]]);
  let stdDev_c3 = cv.matFromArray(1, 1, cv.CV_32FC3, [stdDev.data64F[0], stdDev.data64F[1], stdDev.data64F[2]]);
  let means = new cv.Mat(signal.rows, 1, cv.CV_32FC3);
  let stdDevs = new cv.Mat(signal.rows, 1, cv.CV_32FC3);
  cv.repeat(means_c3, signal.rows, 1, means);
  cv.repeat(stdDev_c3, signal.rows, 1, stdDevs);
  cv.subtract(signal, means, signal, t1, -1);
  cv.divide(signal, stdDevs, signal, 1, -1);
  mean.delete(); stdDev.delete(); t1.delete();
  means_c3.delete(); stdDev_c3.delete();
  means.delete(); stdDevs.delete();
}

// Remove trend in signal
function detrend(signal, lambda) {
  let h = cv.Mat.zeros(signal.rows-2, signal.rows, cv.CV_32FC1);
  let i = cv.Mat.eye(signal.rows, signal.rows, cv.CV_32FC1);
  let t1 = cv.Mat.ones(signal.rows-2, 1, cv.CV_32FC1)
  let t2 = cv.matFromArray(signal.rows-2, 1, cv.CV_32FC1,
    new Array(signal.rows-2).fill(-2));
  let t3 = new cv.Mat();
  t1.copyTo(h.diag(0)); t2.copyTo(h.diag(1)); t1.copyTo(h.diag(2));
  cv.gemm(h, h, lambda*lambda, t3, 0, h, cv.GEMM_1_T);
  cv.add(i, h, h, t3, -1);
  cv.invert(h, h, cv.DECOMP_LU);
  cv.subtract(i, h, h, t3, -1);
  let s = new cv.MatVector();
  cv.split(signal, s);
  cv.gemm(h, s.get(0), 1, t3, 0, s.get(0), 0);
  cv.gemm(h, s.get(1), 1, t3, 0, s.get(1), 0);
  cv.gemm(h, s.get(2), 1, t3, 0, s.get(2), 0);
  cv.merge(s, signal);
  h.delete(); i.delete();
  t1.delete(); t2.delete(); t3.delete();
  s.delete();
}

// Moving average on signal
function movingAverage(signal, n, kernelSize) {
  for (var i = 0; i < n; i++) {
    cv.blur(signal, signal, {height: kernelSize, width: 1});
  }
}

function timeToFrequency(signal, magnitude) {
  // Prepare planes
  let planes = new cv.MatVector();
  planes.push_back(signal);
  planes.push_back(new cv.Mat.zeros(signal.rows, 1, cv.CV_32F))
  cv.merge(planes, signal);
  // Fourier transform
  cv.dft(signal, signal, cv.DFT_COMPLEX_OUTPUT);
  if (magnitude) {
    cv.split(signal, planes);
    cv.magnitude(planes.get(0), planes.get(1), signal);
  }
}

function selectGreen(signal) {
  let rgb = new cv.MatVector();
  cv.split(signal, rgb);
  
  // Extract the green channel (index 1)
  let result = rgb.get(1); // Clone to avoid memory leak
  
  // Clean up allocated Mats
  rgb.delete();

  return result;
}

function getFps(timestamps, timeBase=1000) {
  if (Array.isArray(timestamps) && timestamps.length) {
    if (timestamps.length == 1) {
      return DEFAULT_FPS;
    } else {
      let diff = timestamps[timestamps.length-1] - timestamps[0];
      return timestamps.length/diff*timeBase;
    }
  } else {
    return DEFAULT_FPS;
  }
}

function calculateAverage() {
  // Extract numbers and convert them to integers
  const values = [
    getNumberFromElement(foreheadDisplay),
    getNumberFromElement(chinDisplay),
    getNumberFromElement(mouthDisplay),
    getNumberFromElement(rightCheekDisplay),
    getNumberFromElement(leftCheekDisplay),
    getNumberFromElement(noseDisplay),
    getNumberFromElement(nasionDisplay)
  ];
  // Filter out NaN values in case of any invalid number
  const validValues = values.filter(value => !isNaN(value));

  // Calculate the average
  const sum = validValues.reduce((acc, val) => acc + val, 0);
  const average = validValues.length ? sum / FACE_ZONES.length : 0;
  heartrateBuffer.push(average);
  if (heartrateBuffer.length > 100) {
    heartrateBuffer.shift();
  }
  const getAverageHeartrate = (numbers) => numbers.reduce((acc, number) => acc + number, 0) / numbers.length;
  // Display the average
  avirageDisplay.innerText = `average-bpm: ${getAverageHeartrate(heartrateBuffer).toFixed(0)} BPM`;
}
function getNumberFromElement(element) {
  const text = element.textContent;
  const match = text.match(/(\d+(\.\d+)?)/);
  return match ? parseFloat(match[0]) : 0;
}

