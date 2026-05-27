# pose-action-recognition
A real-time human action recognition system using MediaPipe pose keypoints, a PyTorch LSTM, FastAPI, and Gradio.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-ee4c2c)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-00c2a8)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ed)
![MLflow](https://img.shields.io/badge/MLflow-2.12.0-0194e2)

## Overview
This system provides an end-to-end pipeline for real-time human action recognition. It extracts pose keypoints from a video stream using MediaPipe, processes them through a sliding window approach, and predicts the current action using a trained PyTorch LSTM model. The inference engine is exposed via a robust FastAPI backend and can be tested interactively using a Gradio web interface or an OpenCV real-time demo.

## Architecture Diagram
```text
Webcam/Video
     │
     ▼
┌──────────────┐
│  MediaPipe   │ Extract Pose
│  Keypoints   │
└──────────────┘
     │
     ▼
┌──────────────┐
│Sliding Window│ Buffer Sequence
│  (frames)    │
└──────────────┘
     │
     ▼
┌──────────────┐
│ PyTorch LSTM │ Predict Action
│    Model     │
└──────────────┘
     │
     ▼
┌──────────────┐    ┌──────────────┐
│   FastAPI    │───▶│ OpenCV Overlay│
│  Inference   │    │  (Demo App)  │
└──────────────┘    └──────────────┘
```

## Actions Recognized
| Action | Description |
| :--- | :--- |
| **push_up** | Distinguishing motion in shoulders, elbows, and wrists moving up and down in plank position. |
| **squat** | Distinguishing motion in hips, knees, and ankles with vertical body translation. |
| **wave** | Distinguishing repetitive side-to-side motion of an arm/wrist above shoulder level. |
| **jumping_jack** | Distinguishing synchronized outward and inward motion of both arms and legs. |
| **idle** | Minimal to no distinguishing motion; standing or sitting still. |

## Tech Stack
| Layer | Tool | Purpose |
| :--- | :--- | :--- |
| Pose Extraction | MediaPipe | Extract 3D skeletal keypoints from video frames in real-time. |
| Model Architecture | PyTorch | Implement and train the LSTM sequence classification model. |
| API Backend | FastAPI | Serve the trained model via high-performance REST endpoints. |
| Web UI | Gradio | Provide a user-friendly web interface for uploading videos and testing. |
| Tracking | MLflow | Track training runs, metrics, and manage model versions. |
| Containerization | Docker | Package the API server for consistent deployment. |

## Project Structure
```text
pose-action-recognition/
├── data/
│   └── sequences/              # Empty directory, with .gitkeep
├── models/                     # Empty directory, with .gitkeep
├── app/
│   ├── __init__.py             # App package initialization
│   ├── main.py                 # FastAPI application entrypoint
│   ├── inference.py            # Inference engine logic
│   └── model.py                # PyTorch LSTM model architecture definition
├── train/
│   ├── __init__.py             # Train package initialization
│   ├── dataset.py              # PyTorch Dataset for loading sequences
│   └── train.py                # Training loop with MLflow integration
├── data_collector.py           # Script for webcam keypoint collection
├── realtime_demo.py            # OpenCV real-time inference demonstration
├── gradio_app.py               # Gradio web interface demo
├── mlflow_utils.py             # MLflow helper functions
├── requirements.txt            # Local/CPU Python dependencies
├── requirements_colab.txt      # Colab/GPU training dependencies
├── Dockerfile                  # Container definition for FastAPI server
├── docker-compose.yml          # Orchestration for API and MLflow
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/anantha037/pose-action-recognition.git
cd pose-action-recognition
```

**2. Create a virtual environment**
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

**3. Install dependencies**
> **Note on PyTorch (CPU-only):** This installation uses the CPU-only PyTorch wheel to save space locally, as the CUDA build is ~2.5 GB and unnecessary for inference/data collection.
```bash
# The requirements.txt file contains the extra-index-url for CPU PyTorch
pip install -r requirements.txt
```

## Usage

### (a) Data Collection
Collect pose keypoints using your webcam to build the dataset.
```bash
python data_collector.py --action push_up --samples 50
```

### (b) Training on Colab
Upload the repository to Google Drive or Colab. Do not reinstall torch.
```bash
# In a Colab cell
!pip install -r requirements_colab.txt
!python train/train.py --epochs 50 --batch_size 32
```

### (c) Running FastAPI Server
Start the local inference server.
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### (d) Real-time Demo
Run the OpenCV webcam application.
```bash
python realtime_demo.py
```

### (e) Gradio UI
Launch the interactive web demo.
```bash
python gradio_app.py
```

## MLflow Tracking
This project uses MLflow to track model hyper-parameters, training metrics (accuracy, loss), and model artifacts. To launch the MLflow UI and view the logs locally:
```bash
mlflow ui --host 127.0.0.1 --port 5000
```

## Docker
Build and run the FastAPI inference API via Docker. The image uses CPU PyTorch.
```bash
# Start the API and MLflow server
docker-compose up --build -d

# Check API status
curl http://localhost:8000/docs
```

## Hardware Notes
The local inference, data collection, and serving components are heavily optimized for CPU-only environments (e.g., standard laptops). Utilizing the CPU-only version of PyTorch ensures a lightweight setup without sacrificing real-time inference speed, as the LSTM combined with MediaPipe is efficient enough to run at 30+ FPS on a standard Intel Core i5 processor.

## Author
**Anantha Krishnan**
- GitHub: [https://github.com/anantha037](https://github.com/anantha037)
