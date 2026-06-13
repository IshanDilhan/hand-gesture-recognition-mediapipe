# 🤖 High-Accuracy HRI Gesture Recognition Pipeline

An ultra-robust, real-time hand gesture recognition system designed for **Human-Robot Interaction (HRI)** scenarios, optimized for deployment on resource-constrained platforms like the **NVIDIA Jetson Orin Nano**.

This system implements a hybrid pipeline:
1. **MediaPipe Hands**: Extracts 21 skeletal joints (3D landmarks) per hand in real-time.
2. **Keras/TensorFlow Lite Static MLP Classifier**: A super lightweight, quantized neural network (~6-8 KB) trained on a custom mapped HaGRID dataset (6 classes).
3. **Temporal Motion Resolver**: A rolling history buffer tracks trajectory over time to resolve dynamic gestures like waving or beckoning.

---

## 🌟 Key Features

*   **Jetson Orin Nano Ready**: Highly optimized, quantized TFLite inference takes <0.1 ms.
*   **6 Static Hand Poses**: Model-based detection of Open Palm, Close, Pointer, Thumbs Up, Thumbs Down, and Beckoning.
*   **Landmark Smoothing Filter**: Built-in Exponential Moving Average (EMA) filter ($\alpha = 0.45$) stops coordinates from jittering ("dancing points"), stabilizing both display skeletal lines and model inputs.
*   **Rotation-Invariant Pointing**: Trained using 360-degree rotation data augmentation. Pointing works in all directions (sideways, downwards, diagonally).
*   **EXIF Orientation Metadata Auto-Rotation**: Automatically rotates phone-recorded videos (90°, 180°, 270°) to an upright orientation before MediaPipe processing.
*   **Aspect Ratio Resizing**: Scales high-resolution frames (e.g. 4K) to a maximum dimension of 960px to prevent window clipping and increase execution speed (FPS) significantly.
*   **Flex-Pose Hand Raising**: Identifies "One Hand Raised" or "Arms Up" with open palms, fists, or unknown shapes using dynamic vertical coordinate history tracking (monitoring start and end positions).
*   **Scale-Invariant Waving & Beckoning**: Normalizes coordinate excursions by the hand's own scale, enabling movement triggers (circular, waving, curling) at any distance from the camera.
*   **Menu-Driven Video Playlist Player**: Automatically plays test videos sequentially, loops back, and runs HRI scenario checks.

---

## 📊 Mapped HRI Gestures

The system classifies hand landmarks into 6 static categories:

| ID | Pose Name | HaGRID Folders Mapped | HRI Scenario Intent |
|---|---|---|---|
| 0 | **Open Palm** | `train_val_palm`, `train_val_stop` | `raise hand`, `wave` (static), `both hands up` |
| 1 | **Close (Fist)** | `train_val_fist` | Neutral resting hand state |
| 2 | **Pointer** | `train_val_one` | `point` (stirring, writing, floor mess, pointing) |
| 3 | **Thumbs Up** | `train_val_like` | `thumbs up` (confirm, task success, sarcasm) |
| 4 | **Thumbs Down** | `train_val_dislike` | `thumbs down` (help request, failure, break) |
| 5 | **Beckoning** | `train_val_call` | `beckoning` (static pose shape) |

---

## 📂 Step-by-Step Documentation (A to Z)

We have created individual guides detailing every step of the pipeline under the `doc/` directory:

1.  [doc/step_1_data_mapping.md](file:///d:/FYP/FYP_Motion%20&%20Gesture/Gesture_final/doc/step_1_data_mapping.md): Details the HaGRID folder-to-intent mappings.
2.  [doc/step_2_extraction.md](file:///d:/FYP/FYP_Motion%20&%20Gesture/Gesture_final/doc/step_2_extraction.md): Explains raw landmark calculation, translation, and scale-normalization.
3.  [doc/step_3_training.md](file:///d:/FYP/FYP_Motion%20&%20Gesture/Gesture_final/doc/step_3_training.md): Covers MLP model training, parameters, and rotation augmentation.
4.  [doc/step_4_quantization.md](file:///d:/FYP/FYP_Motion%20&%20Gesture/Gesture_final/doc/step_4_quantization.md): Explains Keras weights-to-TFLite quantization and optimization benefits.
5.  [doc/step_5_realtime_inference.md](file:///d:/FYP/FYP_Motion%20&%20Gesture/Gesture_final/doc/step_5_realtime_inference.md): Explains Exponential Moving Average (EMA) landmark smoothing and temporal resolutions.
6.  [doc/step_6_jetson_deployment.md](file:///d:/FYP/FYP_Motion%20&%20Gesture/Gesture_final/doc/step_6_jetson_deployment.md): Contains instructions, python environment commands, and tips for Jetson Orin Nano deployment.
7.  [doc/model_classifiers_guide.md](file:///d:/FYP/FYP_Motion%20&%20Gesture/Gesture_final/doc/model_classifiers_guide.md): Describes the Keypoint and Point History models and their files.

---

## 🏃 Running the Application

### 1. Feature Extraction
Run MediaPipe landmark extraction on the HaGRID image folders to generate `keypoint.csv`:
```bash
env\Scripts\python extract_dataset.py --max_per_class 1000
```

### 2. Model Training
Train the 6-class MLP model and convert it to a quantized TFLite model:
```bash
env\Scripts\python train/train_keypoint.py
```
This saves the model weights to `model/keypoint_classifier/keypoint_classifier.tflite`.

### 3. Running Real-time Inference (Webcam)
```bash
env\Scripts\python app.py
```

### 4. Running Video Playlist Test
```bash
env\Scripts\python play_video.py
```

---

## 📁 Repository Structure

```
Gesture_final/
├── app.py                          # Live webcam HRI gesture detection
├── play_video.py                   # Playlist video player (TFLite)
├── test_video.py                   # Video sequence tester (TFLite)
├── extract_dataset.py              # Script to extract landmarks from HaGRID images
├── doc/                            # Step-by-step pipeline guides
├── train/
│   ├── train_keypoint.py           # Command-line model training script
│   └── keypoint_classification.ipynb # Jupyter notebook for training
├── model/
│   ├── keypoint_classifier/        # Keypoint pose classifier weights & datasets
│   └── point_history_classifier/   # Point history temporal classifier weights
└── testVideo/                      # Sample HRI scenario test videos
```
