# Smart Vision AI – Real-Time Motion & Human Target Analysis

Smart Vision AI is an advanced real-time computer vision system designed to detect motion, analyze targets, and identify human presence using artificial intelligence.

The system combines motion detection, temporal frame buffering, bidirectional frame analysis, and MediaPipe pose recognition to intelligently isolate and evaluate objects appearing in the camera feed.

Once movement is detected, the system temporarily halts the live scanner and launches a high-speed AI analysis engine that evaluates the captured frames forward and backward to determine the most informative frame containing the target.

The application is built with Python, OpenCV, MediaPipe, and PyQt5, providing both real-time visual feedback and an interactive GUI.

---

## Key Features

- Real-time motion detection using background subtraction
- Debounced motion confirmation to reduce false positives
- Temporal frame buffering for retrospective analysis
- AI-assisted human pose detection with MediaPipe
- Bidirectional frame scanning (Forward + Reverse analysis)
- Advanced edge detection and visual density scoring
- Dynamic target scoring system to locate the best frame
- Dual-view interface (Global Scene + AI Target Zoom)
- Interactive GUI built with PyQt5
- Automatic environment re-scan after analysis

---

## System Architecture

The system is composed of three primary components:

### 1. Environment Scanner

A continuous camera scanner responsible for:

- Capturing live frames from the camera
- Detecting movement using background subtraction
- Confirming motion with debounce logic
- Buffering recent frames for later analysis

Once confirmed motion is detected, the scanner pauses and passes buffered frames to the analysis engine.

---

### 2. AI Target Analysis Engine

This module performs deep frame analysis using multiple techniques:

- Bidirectional temporal scanning
- Edge density measurement
- Region scoring based on spatial information
- Human detection using MediaPipe Pose landmarks

Frames are processed in two directions:

Forward Scan: Oldest → Newest

Backward Scan: Newest → Oldest

Each frame receives a score calculated from:

Score = Edge Density + Target Area + Human Detection Bonus

The frame with the highest score becomes the final detected target.

---

### 3. Visual Interface

The graphical interface provides two synchronized views:

Global View  
Displays the entire scene with the detected region highlighted.

Local View  
Shows a zoomed-in AI processed view of the detected target.

The UI also provides a restart button allowing the system to resume environmental scanning.

---

## Motion Confirmation System

To prevent false detections caused by lighting or noise, the system uses a multi-stage debounce process:

1. Motion candidate detection using contour analysis
2. Region color histogram comparison using Bhattacharyya distance
3. Temporal validation across multiple frames
4. Motion confirmation using ROI contour activity

Only after consistent motion is verified does the AI analysis stage begin.

---

## Human Detection

Human presence is detected using **MediaPipe Pose estimation**.

When a pose skeleton is detected:

- The bounding box automatically expands to include the full body
- A large score bonus is applied
- The frame becomes highly prioritized in final selection

---

## Target Scoring Algorithm

Each analyzed frame receives a score based on:

Score = Edge Density + Bounding Box Area + Human Bonus

Human detections receive a large priority bonus to ensure correct target prioritization.

---

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- PyQt5

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/smart-vision-ai.git
cd smart-vision-ai
```

Install dependencies:

```bash
pip install opencv-python mediapipe numpy pyqt5
```

---

## Usage

Run the application:

```bash
python main.py
```

The system will start in **Standby Mode**, continuously scanning the environment.

When motion is detected:

1. Live scanning pauses
2. AI analysis begins
3. The best frame containing the target is identified
4. The result is displayed on screen

Press **RE-SCAN ENVIRONMENT** to restart scanning.

---

## Project Structure

```
smart-vision-ai/
│
├── main.py
├── README.md
└── LICENSE
```

---

## Future Improvements

Possible enhancements include:

- GPU-accelerated processing
- Multi-object tracking
- Face recognition integration
- Threat classification models
- Remote monitoring dashboard
- Video recording and playback

---

## Disclaimer

This project is intended for educational and research purposes in computer vision and real-time motion analysis.

---

## License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.
