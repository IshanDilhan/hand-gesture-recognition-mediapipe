# 🤖 Gesture_final_accurate — High-Accuracy HRI Gesture Recognition

An ultra-robust, real-time hand gesture recognition system specifically designed for **Human-Robot Interaction (HRI)** scenarios. 

This repository implements a high-performance **rule-based (heuristic) landmark classification pipeline** that achieves near-perfect accuracy and zero-lag CPU processing. It leverages Google's **MediaPipe Hands** for 21-joint skeletal extraction and resolves typical camera domain shift issues (e.g. lighting, background noise, camera models) that neural network classifiers often struggle with.

---

## 🌟 Key Features

*   **Near-Perfect Precision:** Employs a deterministic, rule-based finger joint tracking system that ignores environmental noise.
*   **Stable Spatial Hand Tracking:** Uses a distance-based tracking algorithm to preserve unique hand IDs (Slot `0` vs Slot `1`) during fast waves, avoiding typical MediaPipe coordinate swapping issues.
*   **Native Camera Resolution:** Configured to run with native camera aspect ratios to avoid any squashing or finger joint coordinate distortion.
*   **Continuous Video Player Loop:** An interactive selector menu that lists your test videos, automatically loops back upon ending, and runs sequential "Play All" runs without needing manual user inputs.

---

## ✅ 8 Supported HRI Gestures

The system maps skeletal joints to the following 8 target scenarios critical for robotic command execution:

| # | Gesture Name | Type | Skeletal Tracking Logic |
|---|--------------|------|-------------------------|
| 1 | **One Hand Raised** | Static | Open hand high in the frame (y < 0.45), stationary |
| 2 | **Brief Wave** | Dynamic | Short horizontal wrist motion (1 direction change) |
| 3 | **Pointing** | Static | Only index finger extended, middle/ring/pinky curled |
| 4 | **Beckoning** | Dynamic | Vertical index joint bending / "come here" motion |
| 5 | **Wave** | Dynamic | Sustained horizontal wrist motion (multiple direction changes) |
| 6 | **Arms Up** | Static | Both hands open and held high in the frame (y < 0.42) |
| 7 | **Arms Waving** | Dynamic | Both hands simultaneously executing horizontal waving |
| 8 | **None** | Static | Resting posture, closed fist, or unrecognized hand state |

---

## 🚀 Installation & Setup (Windows)

To set up the clean, isolated virtual environment, follow these steps in your PowerShell terminal:

### 1. Initialize Virtual Environment
```powershell
# Create environment
python -m venv env

# Activate environment
.\env\Scripts\activate
```

### 2. Install High-Performance Dependencies
```powershell
pip install opencv-python==4.9.0.80 mediapipe==0.10.11 tensorflow==2.15.1 numpy==1.26.4
```

---

## 🏃 Running the Application

Ensure your virtual environment is active, or invoke python directly from the environment folder:

### 1. Live Camera Detection (Webcam)
Runs real-time HRI gesture detection through your webcam with the native camera aspect ratio and full MediaPipe skeletal tracking model complexity.
```powershell
.\env\Scripts\python.exe app.py
```
*Press `ESC` or `q` to close the webcam window.*

### 2. Interactive Video Playlist Player
Displays a command-line interface listing all the videos inside your `testVideo/` folder and lets you run detection on them:
```powershell
.\env\Scripts\python.exe play_video.py
```
*   **Select Video Number:** Plays that specific video and automatically returns to the menu when done.
*   **Select `0` (Play All):** Sequentially plays all videos in the directory back-to-back, transitioning automatically with no manual keys required.
*   **Controls during playback:**
    *   `SPACE` — Pause / Resume
    *   `ESC` — Instantly abort playback and return to the main menu
    *   `q` (in terminal menu) — Safely exit the program

---

## 📁 Repository Structure

```
Gesture_final_accurate/
├── .gitignore                      # Configured to exclude env & large video binaries
├── app.py                          # Live webcam HRI gesture detection
├── play_video.py                   # Automatic looping playlist video player
├── utils/
│   ├── __init__.py
│   └── cvfpscalc.py                # Ultra-smooth FPS estimation buffer
└── testVideo/
    └── .gitkeep                    # Folder placeholder (video files are gitignored)
```

---

## 🛠️ Heuristic Skeletal Logic (Under the Hood)

Instead of passing highly noisy raw pixels into a standard neural network (which frequently makes errors under poor lighting or on diverse hand sizes), the system uses a clean mathematical model:

1. **Finger Extension Check:**
   $$\text{Finger Up} = \text{Landmark(Tip)}_y < \text{Landmark(MCP)}_y$$
2. **Wrist Movement Excursion:**
   Tracks the coordinate bounds $[x_{\min}, x_{\max}]$ over a sliding buffer window of 25 frames. If the maximum distance exceeds `0.14` and has at least two direction changes ($\text{sign}(\Delta x)$ reversals), a sustained **Wave** is triggered.
3. **Beckoning Velocity:**
   Monitors the vertical difference between the index finger tip and PIP joints over time.
