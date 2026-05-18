# Gesture_final_accurate — HRI Gesture Recognition System
# ========================================================
# High-accuracy MediaPipe + TFLite MLP gesture recognition for service robots.
#
# 8 Supported Gestures (No Reaching):
#   Static:  One hand raised, Pointing, Arms up, None
#   Dynamic: Brief wave, Wave, Arms waving, Beckoning
#
# ## Quick Start
#
# ### Step 1: Install Dependencies
#   pip install opencv-python==4.9.0.80 mediapipe==0.10.11 tensorflow==2.15.1 numpy==1.26.4
#
# ### Step 2: Extract Training Data (Optional - already have trained models)
#   python extract_dataset.py --max_per_class 1500
#
# ### Step 3: Train Model (Optional - open in Jupyter)
#   jupyter notebook train/keypoint_classification.ipynb
#
# ### Step 4: Run Real-time Detection
#   python app.py
#
# ### Step 5: Test with Video
#   python test_video.py --video testVideo/test.mp4
#
# ## Project Structure
#
#   Gesture_final_accurate/
#   ├── app.py                         # Real-time gesture detection (camera)
#   ├── test_video.py                  # Video-based gesture detection
#   ├── extract_dataset.py             # HaGRID → keypoint.csv extraction
#   ├── model/
#   │   ├── __init__.py
#   │   ├── keypoint_classifier/
#   │   │   ├── keypoint_classifier.py
#   │   │   ├── keypoint_classifier.tflite    # Trained static gesture model
#   │   │   ├── keypoint_classifier.hdf5      # Keras backup
#   │   │   ├── keypoint_classifier_label.csv # Open, Close, Pointer, OK
#   │   │   └── keypoint.csv                  # Training data
#   │   └── point_history_classifier/
#   │       ├── point_history_classifier.py
#   │       ├── point_history_classifier.tflite # Trained dynamic gesture model
#   │       ├── point_history_classifier.hdf5   # Keras backup
#   │       ├── point_history_classifier_label.csv # Stop, Clockwise, etc.
#   │       └── point_history.csv               # Training data
#   ├── train/
#   │   ├── keypoint_classification.ipynb       # Training notebook
#   │   └── point_history_classification.ipynb
#   ├── utils/
#   │   ├── __init__.py
#   │   └── cvfpscalc.py
#   └── README.py
#
# ## Pipeline
#   Camera → MediaPipe Hands (21 landmarks) → Normalize → TFLite MLP → Scenario Resolution
#
# ## Keyboard Shortcuts (during app.py)
#   n: Normal mode
#   k: Keypoint logging mode (press 0-9 to save samples)
#   h: History logging mode (press 0-9 to save samples)
#   ESC: Quit
