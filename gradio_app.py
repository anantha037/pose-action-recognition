import os
import time
import tempfile
import collections
from typing import Tuple, Dict, Any, List

import cv2
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

from app.inference import ActionClassifier

# Global model initialization
MODEL_PATH = "models/best_lstm.pt"
classifier = None

def load_model() -> None:
    """Loads the ActionClassifier globally."""
    global classifier
    if os.path.exists(MODEL_PATH):
        try:
            classifier = ActionClassifier(model_path=MODEL_PATH, device="cpu")
            print(f"Successfully loaded model from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            classifier = None
    else:
        print(f"Model file not found at {MODEL_PATH}")

load_model()

# Action colors (BGR for OpenCV, Hex for matplotlib)
# 6 action classes: idle, jumping_jack, push_up, sitting, squat, wave
ACTION_COLORS_BGR = {
    "idle": (150, 150, 150),          # Gray
    "wave": (0, 255, 255),            # Yellow
    "squat": (255, 100, 0),           # Blue
    "push_up": (0, 255, 0),           # Green
    "jumping_jack": (255, 0, 255),    # Magenta
    "sitting": (0, 165, 255),         # Orange
    "low_confidence": (0, 0, 255)     # Red
}

ACTION_COLORS_HEX = {
    "idle": "#969696",
    "wave": "#FFFF00",
    "squat": "#0064FF",
    "push_up": "#00FF00",
    "jumping_jack": "#FF00FF",
    "sitting": "#FFA500",
    "low_confidence": "#FF0000"
}


def extract_landmarks(results: Any) -> List[float]:
    """
    Extracts a flat list of 132 features (x, y, z, visibility for 33 landmarks)
    from MediaPipe pose results.
    """
    if results.pose_landmarks and len(results.pose_landmarks.landmark) > 0:
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        return landmarks
    return [0.0] * 132

def draw_overlay(
    frame: np.ndarray, 
    action: str, 
    confidence: float, 
    threshold: float
) -> None:
    """Draws the semi-transparent top bar and action text on the frame."""
    h, w, _ = frame.shape
    
    # Draw Top Bar (80px height)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    if confidence < threshold and action != "Waiting...":
        display_text = "LOW CONFIDENCE"
        text_color = ACTION_COLORS_BGR["low_confidence"]
    else:
        display_text = action.upper().replace('_', ' ')
        text_color = ACTION_COLORS_BGR.get(action.lower(), (255, 255, 255))
        
    conf_text = f"{confidence * 100:.1f}%" if action != "Waiting..." else ""

    # Action Text
    cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)
    
    # Confidence Text
    if conf_text:
        text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        cv2.putText(frame, conf_text, (w - text_size[0] - 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def create_confidence_plot(
    timeline_data: List[Dict[str, Any]], 
    threshold: float
) -> plt.Figure:
    """Creates a matplotlib figure showing the confidence timeline."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if not timeline_data:
        ax.text(0.5, 0.5, "No predictions made", ha="center", va="center")
        return fig
        
    frames = [item["frame"] for item in timeline_data]
    confidences = [item["confidence"] for item in timeline_data]
    actions = [item["action"] for item in timeline_data]
    
    # Plot segments colored by action
    for i in range(len(frames) - 1):
        action = actions[i]
        color = ACTION_COLORS_HEX.get(action.lower(), "#FFFFFF")
        ax.plot([frames[i], frames[i+1]], [confidences[i], confidences[i+1]], color=color, linewidth=2)
        
    ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label=f"Threshold ({threshold})")
    
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Prediction Confidence Timeline")
    ax.grid(True, alpha=0.3)
    
    # Create custom legend
    unique_actions = set(actions)
    legend_elements = [
        mpatches.Patch(color=ACTION_COLORS_HEX.get(a.lower(), "#FFFFFF"), label=a)
        for a in unique_actions
    ]
    legend_elements.append(plt.Line2D([0], [0], color='r', linestyle='--', label='Threshold'))
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    return fig


def process_video(
    video_path: str, 
    confidence_threshold: float, 
    stride: int,
    progress=gr.Progress()
) -> Tuple[str, Dict[str, Any], plt.Figure]:
    """
    Processes the uploaded video frame by frame, runs pose extraction and action recognition.
    Generates an annotated output video, summary dictionary, and confidence timeline plot.
    """
    if not classifier:
        raise gr.Error("Model is not loaded. Please ensure the model file exists.")
        
    if not video_path:
        raise gr.Error("No video uploaded.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("Failed to open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0 or fps != fps: # Handle nan/0
        fps = 30.0
        
    temp_out = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    out_path = temp_out.name
    temp_out.close()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    pose = mp_pose.Pose(
        model_complexity=0, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    
    buffer = collections.deque(maxlen=30)
    
    frames_processed = 0
    predictions_made = 0
    start_time = time.time()
    
    current_action = "Waiting..."
    current_confidence = 0.0
    
    timeline_data = []
    action_counts = collections.defaultdict(int)
    total_confidence = 0.0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frames_processed += 1
            if frames_processed % 50 == 0:
                progress(frames_processed / total_frames, desc=f"Processing frame {frames_processed}/{total_frames}")
                
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            # Extract features
            features = extract_landmarks(results)
            buffer.append(features)
            
            # Draw MediaPipe skeleton
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS
                )
                
            # Run prediction every 'stride' frames if buffer is full
            if len(buffer) == 30 and (frames_processed % stride) == 0:
                sequence = np.array(buffer)
                pred_result = classifier.predict(sequence)
                
                current_action = pred_result.get("action", "Unknown")
                current_confidence = pred_result.get("confidence", 0.0)
                
                predictions_made += 1
                total_confidence += current_confidence
                action_counts[current_action] += 1
                
                timeline_data.append({
                    "frame": frames_processed,
                    "action": current_action,
                    "confidence": current_confidence
                })
                
            # Draw UI overlay
            draw_overlay(frame, current_action, current_confidence, confidence_threshold)
            
            # Write frame to output video
            out.write(frame)
            
    finally:
        # Cleanup resources
        cap.release()
        out.release()
        pose.close()
        
    processing_time = time.time() - start_time
    
    # Calculate summary metrics
    avg_conf = total_confidence / predictions_made if predictions_made > 0 else 0.0
    dominant_action = max(action_counts.items(), key=lambda x: x[1])[0] if action_counts else "None"
    
    action_distribution = {}
    for act, count in action_counts.items():
        percentage = (count / predictions_made) * 100
        action_distribution[act] = f"{percentage:.1f}%"
        
    summary_dict = {
        "total_frames": frames_processed,
        "predictions_made": predictions_made,
        "dominant_action": dominant_action,
        "average_confidence": round(avg_conf, 3),
        "action_distribution": action_distribution,
        "processing_time_seconds": round(processing_time, 2)
    }
    
    fig = create_confidence_plot(timeline_data, confidence_threshold)
    
    return out_path, summary_dict, fig


def build_app() -> gr.Blocks:
    """Builds and configures the Gradio user interface."""
    with gr.Blocks(theme=gr.themes.Soft(), title="Pose Action Recognition") as demo:
        
        # Display warning if model is missing
        if classifier is None:
            gr.Warning(f"Model file not found at {MODEL_PATH}. Prediction features will not work.")
            
        gr.Markdown("# Pose Action Recognition")
        gr.Markdown("Upload a video to recognize actions using MediaPipe and LSTM.")
        
        with gr.Tabs():
            with gr.TabItem("Video Analysis"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="Upload Action Video")
                        conf_slider = gr.Slider(minimum=0.3, maximum=1.0, value=0.6, step=0.05, label="Confidence Threshold")
                        stride_slider = gr.Slider(minimum=5, maximum=30, value=15, step=5, label="Prediction Stride (frames)")
                        analyze_btn = gr.Button("Analyze Video", variant="primary")
                        
                    with gr.Column():
                        video_output = gr.Video(label="Annotated Output")
                        summary_json = gr.JSON(label="Action Summary")
                        timeline_plot = gr.Plot(label="Confidence Timeline")
                        
            with gr.TabItem("Model Info"):
                model_info = {}
                classes = []
                
                try:
                    # Prefer instance method if classifier is instantiated
                    if classifier and hasattr(classifier, 'get_model_info'):
                        model_info = classifier.get_model_info()
                    elif hasattr(ActionClassifier, 'get_model_info'):
                        model_info = ActionClassifier.get_model_info()
                        
                    # Populate classes if available
                    if classifier and hasattr(classifier, 'classes'):
                        classes = [[i, act] for i, act in enumerate(classifier.classes)]
                    elif hasattr(ActionClassifier, 'classes'):
                        classes = [[i, act] for i, act in enumerate(ActionClassifier.classes)]
                    else:
                        # Default classes fallback
                        default_classes = ["idle", "jumping_jack", "push_up", "sitting", "squat", "wave"]
                        classes = [[i, act] for i, act in enumerate(default_classes)]
                except Exception as e:
                    print(f"Error extracting model info: {e}")
                
                gr.JSON(value=model_info, label="Model Info")
                gr.Dataframe(value=classes, headers=["Index", "Action"], label="Supported Classes")
                
        # Interactive logic
        analyze_btn.click(
            fn=lambda: gr.update(interactive=False),
            outputs=analyze_btn
        ).then(
            fn=process_video,
            inputs=[video_input, conf_slider, stride_slider],
            outputs=[video_output, summary_json, timeline_plot]
        ).then(
            fn=lambda: gr.update(interactive=True),
            outputs=analyze_btn
        )

    return demo

if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
