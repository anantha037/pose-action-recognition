"""
Webcam keypoint collection script.
Uses MediaPipe to extract pose keypoints from webcam feed and saves them as numpy sequences.
"""

import argparse
import os
import time
import cv2
import numpy as np
import mediapipe as mp
from typing import Any


def extract_keypoints(results: Any) -> np.ndarray:
    """
    Extracts 33 pose landmarks from MediaPipe results and flattens them into a 1D array.
    Each landmark has 4 values: x, y, z, visibility.
    If no landmarks are detected, returns a zero array of shape (132,).

    Args:
        results: Output from MediaPipe Pose model.

    Returns:
        np.ndarray: Flattened array of shape (132,) containing keypoints or zeros.
    """
    if results.pose_landmarks:
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return np.array(keypoints, dtype=np.float32)
    else:
        return np.zeros(132, dtype=np.float32)


def get_next_sequence_index(action_dir: str) -> int:
    """
    Finds the next available sequence index by checking existing files in the action directory.

    Args:
        action_dir (str): Directory containing saved sequence files.

    Returns:
        int: The next available sequence index.
    """
    if not os.path.exists(action_dir):
        return 0
    
    existing_files = [f for f in os.listdir(action_dir) if f.endswith('.npy')]
    if not existing_files:
        return 0
    
    indices = []
    for file in existing_files:
        try:
            index_str = file.split('.')[0]
            indices.append(int(index_str))
        except ValueError:
            pass
            
    if not indices:
        return 0
        
    return max(indices) + 1


def draw_overlay(
    image: np.ndarray, 
    action: str, 
    seq_current: int, 
    seq_total: int, 
    frame_current: int, 
    frame_total: int, 
    status: str,
    recording: bool
) -> None:
    """
    Draws information overlay on the given frame.

    Args:
        image (np.ndarray): The current frame from OpenCV.
        action (str): The action being recorded.
        seq_current (int): The current sequence number being recorded.
        seq_total (int): Total sequences to record in this session.
        frame_current (int): Current frame number in the sequence.
        frame_total (int): Total frames per sequence.
        status (str): Current status text (e.g., 'READY', 'RECORDING', 'GET READY...').
        recording (bool): Whether recording is currently active.
    """
    height, width, _ = image.shape
    
    # Draw red border if recording
    if recording:
        border_thickness = 10
        cv2.rectangle(image, (0, 0), (width - 1, height - 1), (0, 0, 255), border_thickness)
        
    # Top-left: Action and Sequence Progress
    info_text = f"Action: {action} | Seq: {seq_current}/{seq_total}"
    cv2.putText(image, info_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Top-right: Frame count
    frame_text = f"Frame: {frame_current}/{frame_total}"
    text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(image, frame_text, (width - text_size[0] - 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Bottom-left: Status
    status_color = (0, 0, 255) if status == "RECORDING" else (0, 255, 0)
    cv2.putText(image, status, (15, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
    
    # Bottom-center: Instruction
    instr_text = "Press Q to quit"
    instr_size = cv2.getTextSize(instr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.putText(image, instr_text, ((width - instr_size[0]) // 2, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def collect_data(action: str, num_sequences: int, sequence_length: int, output_dir: str, camera_index: int) -> None:
    """
    Main function to capture video, process pose, and save sequence data.

    Args:
        action (str): Label name for the action.
        num_sequences (int): Number of sequences to collect.
        sequence_length (int): Frames per sequence.
        output_dir (str): Directory to save output data.
        camera_index (int): Index of the webcam.
    """
    action_dir = os.path.join(output_dir, action)
    os.makedirs(action_dir, exist_ok=True)
    
    start_index = get_next_sequence_index(action_dir)
    print(f"Starting collection for '{action}' at sequence index {start_index}")
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {camera_index}")
        return

    sequences_saved = 0
    current_seq_index = start_index

    try:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            for seq in range(num_sequences):
                # 1. Countdown Phase
                for count in [3, 2, 1]:
                    start_time = time.time()
                    while time.time() - start_time < 1.0:
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False
                        results = pose.process(image)
                        
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        
                        if results.pose_landmarks:
                            mp_drawing.draw_landmarks(
                                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                            )
                        
                        draw_overlay(
                            image=image,
                            action=action,
                            seq_current=seq + 1,
                            seq_total=num_sequences,
                            frame_current=0,
                            frame_total=sequence_length,
                            status=f"Get Ready... {count}",
                            recording=False
                        )
                        
                        cv2.imshow('Pose Data Collection', image)
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            print("Collection interrupted by user.")
                            return
                
                # 2. Recording Phase
                sequence_data = []
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = pose.process(image)

                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                        )
                        
                    keypoints = extract_keypoints(results)
                    sequence_data.append(keypoints)

                    draw_overlay(
                        image=image,
                        action=action,
                        seq_current=seq + 1,
                        seq_total=num_sequences,
                        frame_current=frame_num + 1,
                        frame_total=sequence_length,
                        status="RECORDING",
                        recording=True
                    )
                    
                    cv2.imshow('Pose Data Collection', image)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        print("Collection interrupted by user during recording.")
                        return
                        
                # 3. Save sequence
                np_sequence = np.array(sequence_data)
                save_path = os.path.join(action_dir, f"{current_seq_index}.npy")
                np.save(save_path, np_sequence)
                
                print(f"Saved sequence {seq + 1}/{num_sequences} for action '{action}'")
                current_seq_index += 1
                sequences_saved += 1
                
                # Brief 0.5 second pause between sequences
                pause_start = time.time()
                while time.time() - pause_start < 0.5:
                    ret, frame = cap.read()
                    if ret:
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False
                        results = pose.process(image)
                        
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        
                        if results.pose_landmarks:
                            mp_drawing.draw_landmarks(
                                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                            )
                            
                        draw_overlay(
                            image=image,
                            action=action,
                            seq_current=seq + 1,
                            seq_total=num_sequences,
                            frame_current=sequence_length,
                            frame_total=sequence_length,
                            status="READY",
                            recording=False
                        )
                        cv2.imshow('Pose Data Collection', image)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        print("Collection interrupted by user during pause.")
                        return

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("-" * 50)
        print(f"Collection finished.")
        print(f"Sequences saved in this session: {sequences_saved}")
        
        # Summary of total files
        if os.path.exists(action_dir):
            total_files = len([f for f in os.listdir(action_dir) if f.endswith('.npy')])
            print(f"Total sequences now existing for action '{action}': {total_files}")
        print("-" * 50)


def main() -> None:
    """
    Parses command line arguments and initiates the data collection process.
    """
    parser = argparse.ArgumentParser(description="Collect pose keypoint sequences for action recognition.")
    parser.add_argument("--action", type=str, required=True, help="Label name e.g., push_up")
    parser.add_argument("--sequences", type=int, default=20, help="Number of sequences to collect")
    parser.add_argument("--sequence_length", type=int, default=30, help="Frames per sequence")
    parser.add_argument("--output_dir", type=str, default=os.path.join("data", "sequences"), help="Directory to save .npy files")
    parser.add_argument("--camera_index", type=int, default=0, help="Webcam index for OpenCV")
    
    args = parser.parse_args()
    
    collect_data(
        action=args.action,
        num_sequences=args.sequences,
        sequence_length=args.sequence_length,
        output_dir=args.output_dir,
        camera_index=args.camera_index
    )


if __name__ == "__main__":
    main()
