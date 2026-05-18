#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🤖 HRI Gesture Recognition — Real-time Application (High-Accuracy Webcam Edition)
=================================================================================
Highly robust, real-time hand gesture recognition system designed for HRI.
Uses MediaPipe Hands for skeletal tracking and a precision rule-based classifier.

Optimized for Webcams:
  1. Uses the webcam's native resolution and aspect ratio (prevents distortion).
  2. Uses full-complexity MediaPipe tracking (model_complexity=1) for maximum accuracy.
  3. Implements distance-based stable hand tracking (prevents index swapping during waves).
  4. Matches the highly accurate 'new_gesture' pipeline exactly.
"""

import copy
import argparse
import collections
import time
import cv2 as cv
import numpy as np
import mediapipe as mp
from utils import CvFpsCalc


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="Camera index")
    parser.add_argument("--min_detection_confidence", type=float, default=0.6)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.6)
    return parser.parse_args()


# ── Finger Joint Constants ──────────────────────────────────────────────────
WRIST = 0
INDEX_TIP = 8
INDEX_MCP = 5
MIDDLE_TIP = 12
MIDDLE_MCP = 9
RING_TIP = 16
RING_MCP = 13
PINKY_TIP = 20
PINKY_MCP = 17


# ── Feature Extractors ───────────────────────────────────────────────────────

def finger_up(lm, tip, mcp):
    """Returns True if the finger tip is above its MCP joint (y-axis)."""
    return lm[tip].y < lm[mcp].y


def count_fingers_up(lm):
    """Count open fingers excluding thumb (index, middle, ring, pinky)."""
    fingers = [
        finger_up(lm, INDEX_TIP, INDEX_MCP),
        finger_up(lm, MIDDLE_TIP, MIDDLE_MCP),
        finger_up(lm, RING_TIP, RING_MCP),
        finger_up(lm, PINKY_TIP, PINKY_MCP),
    ]
    return sum(fingers), fingers


def hand_y_norm(lm):
    """Average y of all landmarks. Lower y = hand is higher in the frame."""
    return np.mean([l.y for l in lm])


def classify_single_hand(lm):
    """Accurate rule-based gesture classifier for a single hand."""
    n_up, fingers = count_fingers_up(lm)
    idx_up, mid_up, ring_up, pinky_up = fingers

    # Pointing: only index is up, rest curled
    if idx_up and not mid_up and not ring_up and not pinky_up:
        return "Pointing"

    # OPEN: 4 fingers extended
    if n_up >= 4:
        return "OPEN"

    # Beckoning pre-check: Index extended but curled down
    pip_y = lm[6].y
    tip_y = lm[INDEX_TIP].y
    if idx_up and not mid_up and not ring_up:
        if abs(tip_y - pip_y) < 0.06:
            return "Beckoning"

    return "None"


# ── Stable Distance-Based Motion Tracker ─────────────────────────────────────

class HandMotionTracker:
    def __init__(self):
        # Stably track up to 2 hands using spatial coordinates
        self.x_hist = {0: collections.deque(maxlen=25), 1: collections.deque(maxlen=25)}
        self.beck_hist = {0: collections.deque(maxlen=15), 1: collections.deque(maxlen=15)}
        self.last_positions = {0: None, 1: None}

    def get_track_id(self, wx, wy):
        """Assign ID (0 or 1) based on proximity to previous frame's hand coordinates."""
        # 1. If both slots are empty, assign to 0
        if self.last_positions[0] is None and self.last_positions[1] is None:
            self.last_positions[0] = (wx, wy)
            return 0

        # 2. If only slot 0 is active
        if self.last_positions[0] is not None and self.last_positions[1] is None:
            d0 = (wx - self.last_positions[0][0])**2 + (wy - self.last_positions[0][1])**2
            if d0 < 0.06:  # Proximity threshold
                self.last_positions[0] = (wx, wy)
                return 0
            else:
                self.last_positions[1] = (wx, wy)
                return 1

        # 3. If only slot 1 is active
        if self.last_positions[0] is None and self.last_positions[1] is not None:
            d1 = (wx - self.last_positions[1][0])**2 + (wy - self.last_positions[1][1])**2
            if d1 < 0.06:
                self.last_positions[1] = (wx, wy)
                return 1
            else:
                self.last_positions[0] = (wx, wy)
                return 0

        # 4. If both are active, match the closest one
        d0 = (wx - self.last_positions[0][0])**2 + (wy - self.last_positions[0][1])**2
        d1 = (wx - self.last_positions[1][0])**2 + (wy - self.last_positions[1][1])**2
        if d0 < d1:
            self.last_positions[0] = (wx, wy)
            return 0
        else:
            self.last_positions[1] = (wx, wy)
            return 1

    def update(self, hand_id, wx, beck_angle):
        self.x_hist[hand_id].append(wx)
        self.beck_hist[hand_id].append(beck_angle)

    def is_waving(self, hand_id):
        h = self.x_hist[hand_id]
        if len(h) < 10:
            return False, False

        arr = np.array(h)
        excursion = arr.max() - arr.min()
        diffs = np.diff(arr)
        reversals = int(np.sum(np.diff(np.sign(diffs)) != 0))

        # Horizontal waving rules
        wave = excursion > 0.14 and reversals >= 2
        brief_wave = excursion > 0.08 and reversals >= 1 and len(h) <= 15
        return wave, brief_wave

    def is_beckoning(self, hand_id):
        h = self.beck_hist[hand_id]
        if len(h) < 8:
            return False

        arr = np.array(h)
        excursion = arr.max() - arr.min()
        diffs = np.diff(arr)
        reversals = int(np.sum(np.diff(np.sign(diffs)) != 0))

        return excursion > 0.04 and reversals >= 2

    def reset_inactive(self, active_ids):
        """Clear tracking history for hands that disappeared from the frame."""
        for i in [0, 1]:
            if i not in active_ids:
                self.last_positions[i] = None


# ── Gesture Engine ──────────────────────────────────────────────────────────

class GestureEngine:
    GESTURE_COLORS = {
        "One Hand Raised": (0, 220, 100),
        "Brief Wave": (255, 200, 0),
        "Pointing": (0, 180, 255),
        "None": (160, 160, 160),
        "Arms Waving": (255, 80, 200),
        "Wave": (255, 140, 0),
        "Beckoning": (100, 255, 220),
        "Arms Up": (0, 100, 255),
        "No hands": (80, 80, 80),
    }

    def __init__(self):
        self.tracker = HandMotionTracker()
        self.last_gesture = "None"
        self.last_time = time.time()

    def stable_update(self, gesture):
        now = time.time()
        if gesture != self.last_gesture:
            if now - self.last_time > 0.3:  # 300ms smoothing delay
                self.last_gesture = gesture
                self.last_time = now
        return self.last_gesture

    def process(self, hand_results):
        if not hand_results:
            self.tracker.reset_inactive([])
            return self.stable_update("No hands"), self.GESTURE_COLORS["No hands"]

        n_hands = len(hand_results)
        per_hand = []
        active_ids = []

        for hl in hand_results:
            lm = hl.landmark
            base = classify_single_hand(lm)
            hy = hand_y_norm(lm)
            wx = lm[WRIST].x
            wy = lm[WRIST].y
            beck_a = lm[INDEX_TIP].y - lm[6].y

            # Compute stable spatial ID
            hand_id = self.tracker.get_track_id(wx, wy)
            active_ids.append(hand_id)

            self.tracker.update(hand_id, wx, beck_a)

            per_hand.append({
                'base': base,
                'hy': hy,
                'wx': wx,
                'id': hand_id,
                'lm': lm
            })

        self.tracker.reset_inactive(active_ids)

        # ── Two-Hand Scenarios (Arms Up / Arms Waving) ──────────────────────
        if n_hands == 2:
            h0 = per_hand[0]
            h1 = per_hand[1]

            # Arms Up: Both hands open and held high (y < 0.42)
            both_open = (h0['base'] == "OPEN" and h1['base'] == "OPEN")
            both_high = (h0['hy'] < 0.42 and h1['hy'] < 0.42)
            if both_open and both_high:
                return self.stable_update("Arms Up"), self.GESTURE_COLORS["Arms Up"]

            # Arms Waving: Both hands waving
            w0, bw0 = self.tracker.is_waving(h0['id'])
            w1, bw1 = self.tracker.is_waving(h1['id'])
            if (w0 or bw0) and (w1 or bw1):
                return self.stable_update("Arms Waving"), self.GESTURE_COLORS["Arms Waving"]

        # ── Single-Hand Scenarios (Using dominant hand - the highest hand) ───
        primary = min(per_hand, key=lambda x: x['hy'])
        base = primary['base']
        hy = primary['hy']
        hid = primary['id']

        is_wave, is_brief = self.tracker.is_waving(hid)

        if base == "Beckoning" or self.tracker.is_beckoning(hid):
            return self.stable_update("Beckoning"), self.GESTURE_COLORS["Beckoning"]

        if is_wave and base == "OPEN":
            return self.stable_update("Wave"), self.GESTURE_COLORS["Wave"]

        if is_brief and base == "OPEN":
            return self.stable_update("Brief Wave"), self.GESTURE_COLORS["Brief Wave"]

        if base == "Pointing":
            return self.stable_update("Pointing"), self.GESTURE_COLORS["Pointing"]

        # One Hand Raised: Open hand held high, stationary
        if base == "OPEN" and hy < 0.45:
            return self.stable_update("One Hand Raised"), self.GESTURE_COLORS["One Hand Raised"]

        return self.stable_update("None"), self.GESTURE_COLORS["None"]


# ── Bounding Box Utility ────────────────────────────────────────────────────

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_array = np.append(landmark_array, [[landmark_x, landmark_y]], axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def draw_overlay(frame, label, color):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv.rectangle(overlay, (0, 0), (w, 75), (15, 15, 15), -1)
    cv.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv.putText(frame, "HRI WEBCAM GESTURE DETECTOR  |  ACCURATE",
                (12, 22), cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv.LINE_AA)
    cv.putText(frame, label,
                (12, 62), cv.FONT_HERSHEY_DUPLEX, 1.4, color, 2, cv.LINE_AA)

    # Left color bar indicator
    cv.rectangle(frame, (0, 0), (6, h), color, -1)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    # Match 'new_gesture' behavior: do not force arbitrary camera resolution limits
    # which stretch, crop, or lower frames on various webcams. Let it use its native mode.
    cap = cv.VideoCapture(args.device)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    # Set full complexity (model_complexity=1) for maximum accuracy on webcam
    hands = mp_hands.Hands(
        model_complexity=1,
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    engine = GestureEngine()
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    print("==================================================")
    print("🚀 Running High-Accuracy Webcam Gesture Detector...")
    print("👉 Press ESC or 'q' to exit.")
    print("==================================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process exactly as received (no arbitrary pre-flips to ensure MediaPipe behaves properly)
        fps = cvFpsCalc.get()
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                brect = calc_bounding_rect(frame, hl)
                # Draw skeleton
                mp_draw.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 150), thickness=2, circle_radius=4),
                    mp_draw.DrawingSpec(color=(0, 200, 100), thickness=2),
                )
                # Bounding box
                cv.rectangle(frame, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 1)

            gesture, color = engine.process(res.multi_hand_landmarks)
        else:
            gesture, color = engine.process(None)

        draw_overlay(frame, gesture, color)

        # Draw FPS
        cv.putText(frame, f"FPS: {fps}", (frame.shape[1] - 100, 50),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv.LINE_AA)

        cv.imshow("HRI Hand Gesture Detector (Accurate)", frame)

        key = cv.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    hands.close()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
