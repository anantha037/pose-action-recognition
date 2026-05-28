"""
realtime_demo.py

Real-time webcam demo that extracts MediaPipe pose keypoints, sends sequences to 
the FastAPI server via sliding window, and overlays predictions on the video feed.
"""

import cv2
import time
import argparse
import threading
import collections
import requests
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), 
                    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), 
                    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), 
                    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), 
                    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)]

def get_model_path() -> str:
    """Download the MediaPipe Pose Landmarker model if it doesn't exist."""
    model_path = "pose_landmarker_lite.task"
    if not os.path.exists(model_path):
        print(f"Downloading MediaPipe model to {model_path}...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        urllib.request.urlretrieve(url, model_path)
    return model_path


@dataclass
class DemoState:
    """Holds mutable state for the real-time demo across threads."""
    latest_action: str = "Waiting..."
    latest_confidence: float = 0.0
    latest_probabilities: Dict[str, float] = field(default_factory=dict)
    api_offline: bool = False
    is_predicting: bool = False
    total_predictions: int = 0


def send_prediction_request(state: DemoState, sequence: List[List[float]], api_url: str) -> None:
    """
    Sends a sequence to the FastAPI prediction endpoint in a separate thread.
    
    Args:
        state (DemoState): The mutable application state.
        sequence (List[List[float]]): The 30-frame sequence of 132 features.
        api_url (str): URL of the FastAPI prediction endpoint.
    """
    try:
        payload = {
            "sequence": sequence,
            "return_probabilities": True
        }
        # Use a short timeout so we don't hang threads forever
        response = requests.post(f"{api_url.rstrip('/')}/predict", json=payload, timeout=2.0)
        
        if response.status_code == 200:
            data = response.json()
            state.latest_action = data.get("action", "Unknown")
            state.latest_confidence = data.get("confidence", 0.0)
            state.latest_probabilities = data.get("probabilities", {})
            state.api_offline = False
            state.total_predictions += 1
        else:
            state.api_offline = True
    except requests.RequestException:
        state.api_offline = True
    finally:
        state.is_predicting = False


def extract_landmarks(results: Any) -> List[float]:
    """
    Extracts a flat list of 132 features (x, y, z, visibility for 33 landmarks)
    from MediaPipe pose results.
    """
    if results.pose_landmarks and len(results.pose_landmarks) > 0:
        landmarks = []
        for lm in results.pose_landmarks[0]:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        return landmarks
    return [0.0] * 132


def draw_manual_landmarks(frame: np.ndarray, results: Any) -> None:
    """Draw landmarks manually using cv2."""
    if not results.pose_landmarks or len(results.pose_landmarks) == 0:
        return
        
    h, w, _ = frame.shape
    pose = results.pose_landmarks[0]
    
    # Draw connections
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        lm1 = pose[start_idx]
        lm2 = pose[end_idx]
        if lm1.visibility > 0.1 and lm2.visibility > 0.1:
            pt1 = (int(lm1.x * w), int(lm1.y * h))
            pt2 = (int(lm2.x * w), int(lm2.y * h))
            cv2.line(frame, pt1, pt2, (255, 255, 255), 1)
            
    # Draw points
    for lm in pose:
        if lm.visibility > 0.1:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame, pt, 2, (255, 255, 255), -1)


def get_action_color(action: str) -> Tuple[int, int, int]:
    """
    Returns a BGR color tuple for a given action.
    """
    colors = {
        "idle": (150, 150, 150),          # Gray
        "wave": (0, 255, 255),            # Yellow
        "squat": (255, 100, 0),           # Blue
        "push_up": (0, 255, 0),           # Green
        "jumping_jack": (255, 0, 255)     # Magenta
    }
    return colors.get(action.lower(), (255, 255, 255))


def main() -> None:
    """Main execution loop for the real-time demo."""
    parser = argparse.ArgumentParser(description="Real-time Pose Action Recognition Demo")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000", help="FastAPI server URL")
    parser.add_argument("--camera_index", type=int, default=0, help="Webcam index")
    parser.add_argument("--sequence_length", type=int, default=30, help="Frames per sequence")
    parser.add_argument("--stride", type=int, default=15, help="Frames to slide forward after each prediction")
    parser.add_argument("--confidence_threshold", type=float, default=0.6, help="Minimum confidence to display prediction")
    parser.add_argument("--show_fps", type=bool, default=True, help="Show FPS counter")
    parser.add_argument("--show_probabilities", action="store_true", help="Show per-class probability bars")
    args = parser.parse_args()

    state = DemoState()
    buffer = collections.deque(maxlen=args.sequence_length)
    frames_since_last_prediction = 0
    
    # Initialize Camera
    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera_index}")
        return
    
    print(f"\nStarting real-time demo on camera {args.camera_index}")
    print(f"Connecting to API at {args.api_url}")
    print("Press 'Q' to quit.\n")

    # Initialize MediaPipe Pose with modern Tasks API (avoids 3.13 broken legacy solutions)
    model_path = get_model_path()
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_estimator = vision.PoseLandmarker.create_from_options(options)

    total_frames = 0
    start_time = time.time()
    prev_frame_time = time.time()
    global_timestamp_ms = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            total_frames += 1
            
            # FPS Calculation
            current_time = time.time()
            fps = 1.0 / (current_time - prev_frame_time) if (current_time - prev_frame_time) > 0 else 30.0
            prev_frame_time = current_time

            # MediaPipe extraction (requires RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Use timestamp to maintain tracking across frames
            global_timestamp_ms += max(1, int(1000 / fps))
            results = pose_estimator.detect_for_video(mp_image, global_timestamp_ms)
            
            # Extract features and manage sliding window buffer
            features = extract_landmarks(results)
            buffer.append(features)
            frames_since_last_prediction += 1
            
            # Trigger prediction via thread if buffer is full and stride met
            if len(buffer) == args.sequence_length and frames_since_last_prediction >= args.stride and not state.is_predicting:
                state.is_predicting = True
                frames_since_last_prediction = 0
                
                # We copy the buffer to avoid mutation during thread execution
                seq_copy = list(buffer)
                threading.Thread(
                    target=send_prediction_request, 
                    args=(state, seq_copy, args.api_url), 
                    daemon=True
                ).start()

            # --- Drawing Overlays ---
            
            # 1. Draw Skeleton
            draw_manual_landmarks(frame, results)

            h, w, _ = frame.shape
            
            # 2. Draw Top Bar (80px height)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 80), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            if state.api_offline:
                display_text = "API Offline"
                text_color = (0, 0, 255)  # Red
                conf_text = ""
            else:
                if state.latest_confidence >= args.confidence_threshold:
                    display_text = state.latest_action.upper().replace('_', ' ')
                    text_color = get_action_color(state.latest_action)
                else:
                    display_text = "UNCERTAIN" if state.latest_action != "Waiting..." else "WAITING..."
                    text_color = (200, 200, 200)  # Light gray
                    
                conf_text = f"{state.latest_confidence * 100:.1f}%" if state.latest_action != "Waiting..." else ""

            # Action Text
            cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)
            
            # Confidence Text and Progress Bar
            if conf_text:
                text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                cv2.putText(frame, conf_text, (w - text_size[0] - 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                
                bar_width = int((w - 40) * state.latest_confidence)
                bar_color = (0, 255, 0) if state.latest_confidence >= 0.8 else (0, 255, 255) if state.latest_confidence >= 0.6 else (0, 0, 255)
                cv2.rectangle(frame, (20, 65), (20 + bar_width, 70), bar_color, -1)

            # 3. Draw Bottom Bar (40px height)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 40), (w, h), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            if args.show_fps:
                cv2.putText(frame, f"FPS: {int(fps)}", (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
            buffer_text = f"Buffer: {len(buffer)}/{args.sequence_length}"
            text_size = cv2.getTextSize(buffer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.putText(frame, buffer_text, ((w - text_size[0]) // 2, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            quit_text = "Q - Quit"
            text_size = cv2.getTextSize(quit_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.putText(frame, quit_text, (w - text_size[0] - 20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # 4. Draw Probabilities Panel
            if args.show_probabilities and state.latest_probabilities:
                panel_w = 150
                panel_h = len(state.latest_probabilities) * 30 + 10
                panel_x = w - panel_w - 20
                panel_y = 100
                
                overlay = frame.copy()
                cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                
                y_offset = panel_y + 20
                for action, prob in state.latest_probabilities.items():
                    is_pred = (action == state.latest_action)
                    color = get_action_color(action) if is_pred else (150, 150, 150)
                    text_thickness = 2 if is_pred else 1
                    
                    cv2.putText(frame, f"{action[:8]}: {prob:.2f}", (panel_x + 10, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, text_thickness)
                    
                    bar_w = int(prob * (panel_w - 20))
                    cv2.rectangle(frame, (panel_x + 10, y_offset + 5), (panel_x + 10 + bar_w, y_offset + 8), color, -1)
                    
                    y_offset += 30

            cv2.imshow('Real-time Pose Action Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        total_time = time.time() - start_time
        avg_fps = total_frames / total_time if total_time > 0 else 0
        
        print("\n--- Session Summary ---")
        print(f"Total Frames: {total_frames}")
        print(f"Average FPS:  {avg_fps:.1f}")
        print(f"Total Preds:  {state.total_predictions}")
        print("Shutting down gracefully...")
        
        pose_estimator.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
