"""
video_processor.py

Offline script to process downloaded action videos through MediaPipe and save
keypoint sequences as .npy files. It produces the exact same data format
as data_collector.py but runs autonomously over a directory of videos.
"""

import argparse
import os
import urllib.request
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), 
                    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), 
                    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), 
                    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), 
                    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)]

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process raw videos into keypoint sequences.")
    parser.add_argument("--video_dir", type=str, default="data/raw_videos", 
                        help="Root folder containing per-action subfolders of videos")
    parser.add_argument("--output_dir", type=str, default="data/sequences", 
                        help="Where to save .npy sequence files")
    parser.add_argument("--sequence_length", type=int, default=30, 
                        help="Frames per sequence")
    parser.add_argument("--overlap", type=int, default=15, 
                        help="Sliding window overlap between consecutive sequences")
    parser.add_argument("--min_detection_confidence", type=float, default=0.5, 
                        help="MediaPipe min_detection_confidence")
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5, 
                        help="MediaPipe min_tracking_confidence")
    parser.add_argument("--show_preview", action="store_true", 
                        help="Show an OpenCV window previewing the skeleton overlay frame by frame")
    return parser.parse_args()


def get_model_path() -> str:
    """Download the MediaPipe Pose Landmarker model if it doesn't exist."""
    model_path = "pose_landmarker_lite.task"
    if not os.path.exists(model_path):
        print(f"Downloading MediaPipe model to {model_path}...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        urllib.request.urlretrieve(url, model_path)
    return model_path


def get_next_sequence_index(action_output_dir: Path) -> int:
    """
    Get the next available sequence index for a given action output directory.
    Finds the maximum integer among the existing {idx}.npy files and returns max + 1.
    """
    if not action_output_dir.exists():
        action_output_dir.mkdir(parents=True, exist_ok=True)
        return 0
    
    max_idx = -1
    for filepath in action_output_dir.glob("*.npy"):
        try:
            idx = int(filepath.stem)
            if idx > max_idx:
                max_idx = idx
        except ValueError:
            pass
    return max_idx + 1


def extract_pose_landmarks(results: Any) -> np.ndarray:
    """
    Extracts pose landmarks from a MediaPipe tasks results object.
    
    Returns a numpy array of shape (132,) containing 33 landmarks * 4 values 
    (x, y, z, visibility). If no pose is detected, returns a zero array of shape (132,).
    """
    if results.pose_landmarks and len(results.pose_landmarks) > 0:
        # Flatten the landmarks into a single array
        landmarks = []
        for lm in results.pose_landmarks[0]:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(landmarks, dtype=np.float32)
    else:
        return np.zeros((132,), dtype=np.float32)


def draw_manual_landmarks(frame: np.ndarray, results: Any) -> None:
    """Draw landmarks using cv2 since mp.solutions is unavailable in Tasks API."""
    if not results.pose_landmarks or len(results.pose_landmarks) == 0:
        return
        
    h, w, _ = frame.shape
    pose = results.pose_landmarks[0]
    
    # Draw connections
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        lm1 = pose[start_idx]
        lm2 = pose[end_idx]
        # Only draw if visibility is reasonable
        if lm1.visibility > 0.1 and lm2.visibility > 0.1:
            pt1 = (int(lm1.x * w), int(lm1.y * h))
            pt2 = (int(lm2.x * w), int(lm2.y * h))
            cv2.line(frame, pt1, pt2, (245, 117, 66), 2)
            
    # Draw points
    for lm in pose:
        if lm.visibility > 0.1:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame, pt, 2, (245, 66, 230), -1)


def process_video(
    video_path: Path,
    action: str,
    output_dir: Path,
    pose_estimator: Any,
    sequence_length: int,
    overlap: int,
    show_preview: bool,
    global_seq_idx: int,
    global_timestamp_ms: int
) -> Tuple[int, int, int, int]:
    """
    Process a single video file, extract sequences, and save them.
    
    Returns:
        A tuple of (sequences_extracted, skipped_sequences, next_global_idx, next_global_timestamp_ms).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return 0, 0, global_seq_idx
        
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames_in_video / fps if fps > 0 else 0.0
    
    if fps <= 0:
        fps = 30.0  # Fallback
    
    relative_path = f"{action}/{video_path.name}"
    print(f"Processing: {relative_path}")
    print(f"  Total frames: {total_frames_in_video} | FPS: {fps:.1f} | Duration: {duration:.1f}s")
    
    action_output_dir = output_dir / action
    action_output_dir.mkdir(parents=True, exist_ok=True)
    
    buffer: List[np.ndarray] = []
    sequences_extracted = 0
    skipped_sequences = 0
    current_global_idx = global_seq_idx
    
    step = sequence_length - overlap
    if step <= 0:
        step = 1  # Fallback to prevent infinite loops if overlap is misconfigured
    
    try:
        for frame_idx in tqdm(range(total_frames_in_video), desc="Frames", unit="frame", leave=False):
            success, frame = cap.read()
            if not success:
                break
                
            # Process frame with MediaPipe Tasks API
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Use timestamp to maintain tracking across frames
            global_timestamp_ms += max(1, int(1000 / fps))
            results = pose_estimator.detect_for_video(mp_image, global_timestamp_ms)
            
            # Extract landmarks
            landmarks = extract_pose_landmarks(results)
            buffer.append(landmarks)
            
            # Check if we have a full sequence
            if len(buffer) == sequence_length:
                # Analyze sequence for missing poses
                sequence_array = np.array(buffer)  # Shape: (sequence_length, 132)
                
                # Check for frames where all features are zero (pose not detected)
                zero_frames = np.sum(np.all(sequence_array == 0, axis=1))
                zero_ratio = zero_frames / sequence_length
                
                if zero_ratio > 0.3:
                    # Skip sequence
                    skipped_sequences += 1
                    tqdm.write(f"Warning: Skipped sequence in {video_path.name} "
                               f"(>30% frames lack pose detection)")
                else:
                    # Save sequence
                    save_path = action_output_dir / f"{current_global_idx}.npy"
                    np.save(str(save_path), sequence_array)
                    sequences_extracted += 1
                    current_global_idx += 1
                
                # Slide the window forward by dropping the first 'step' frames
                buffer = buffer[step:]
                
            # Show preview
            if show_preview:
                draw_manual_landmarks(frame, results)
                
                cv2.putText(frame, f"Action: {action}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Frame: {frame_idx + 1}/{total_frames_in_video}", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                cv2.imshow('MediaPipe Pose Preview', frame)
                
                # Press 'q' to skip to next video
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    tqdm.write("\nSkipping to next video...")
                    break
                    
    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()
            
    print(f"  Sequences extracted: {sequences_extracted} | Skipped (low pose detection): {skipped_sequences}")
    return sequences_extracted, skipped_sequences, current_global_idx, global_timestamp_ms


def main() -> None:
    """Main execution point of the script."""
    args = parse_args()
    
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    
    if not video_dir.exists():
        print(f"Error: Video directory '{video_dir}' does not exist.")
        return
        
    # Discover actions (subfolders)
    actions = sorted([d.name for d in video_dir.iterdir() if d.is_dir()])
    if not actions:
        print(f"No action subfolders found in {video_dir}")
        return
        
    video_extensions = {".mp4", ".avi", ".mov"}
    
    # Initialize MediaPipe Pose using the new Tasks API
    model_path = get_model_path()
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=args.min_detection_confidence,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    
    pose_estimator = vision.PoseLandmarker.create_from_options(options)
    
    # Summary stats
    summary: Dict[str, Dict[str, int]] = {action: {"videos": 0, "sequences": 0} for action in actions}
    total_videos = 0
    total_sequences = 0
    global_timestamp_ms = 0
    
    try:
        for action in actions:
            action_dir = video_dir / action
            videos = []
            for ext in video_extensions:
                videos.extend(action_dir.rglob(f"*{ext}"))
                videos.extend(action_dir.rglob(f"*{ext.upper()}"))
                
            # Remove duplicates and sort
            videos = sorted(list(set(videos)))
            
            if not videos:
                continue
                
            summary[action]["videos"] = len(videos)
            total_videos += len(videos)
            
            # Find starting index for this action
            action_output_dir = output_dir / action
            global_seq_idx = get_next_sequence_index(action_output_dir)
            
            for video_path in videos:
                extracted, skipped, global_seq_idx, global_timestamp_ms = process_video(
                    video_path=video_path,
                    action=action,
                    output_dir=output_dir,
                    pose_estimator=pose_estimator,
                    sequence_length=args.sequence_length,
                    overlap=args.overlap,
                    show_preview=args.show_preview,
                    global_seq_idx=global_seq_idx,
                    global_timestamp_ms=global_timestamp_ms
                )
                summary[action]["sequences"] += extracted
                total_sequences += extracted
                
    finally:
        pose_estimator.close()
        
    # Print final summary table
    print("\n" + "─" * 41)
    print(f" {'Action':<15} {'Videos':<9} {'Sequences':<9}")
    print("─" * 41)
    for action in actions:
        if summary[action]["videos"] > 0:
            v_count = summary[action]["videos"]
            s_count = summary[action]["sequences"]
            print(f" {action:<15} {v_count:<9} {s_count:<9}")
    print("─" * 41)
    print(f" {'TOTAL':<15} {total_videos:<9} {total_sequences:<9}")
    print("─" * 41)


if __name__ == "__main__":
    main()
