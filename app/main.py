"""
app/main.py

FastAPI application for serving the pose action recognition model.
Provides REST endpoints for single and batched sequence predictions.
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

import numpy as np
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

# Add parent directory to path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.inference import ActionClassifier


# ---------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request payload for a single sequence prediction."""
    sequence: List[List[float]] = Field(
        ..., 
        description="A 2D array of shape (30, 132) representing keypoints over 30 frames.",
        json_schema_extra={"example": [[0.0] * 132] * 30}
    )
    return_probabilities: bool = Field(
        default=True,
        description="Whether to return full class probabilities alongside the top prediction."
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sequence": [[0.0] * 132] * 30,
                "return_probabilities": True
            }
        }
    )

class PredictResponse(BaseModel):
    """Response payload for a single prediction."""
    action: str = Field(..., description="The predicted action class.")
    confidence: float = Field(..., description="Confidence score for the predicted class [0, 1].")
    probabilities: Optional[Dict[str, float]] = Field(
        default=None, 
        description="Dictionary of all class probabilities."
    )
    latency_ms: float = Field(..., description="Inference latency in milliseconds.")
    sequence_length: int = Field(..., description="Number of frames processed.")
    features_per_frame: int = Field(..., description="Number of features per frame.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "action": "push_up",
                "confidence": 0.9823,
                "probabilities": {"idle": 0.002, "push_up": 0.9823, "jumping_jack": 0.0155, "squat": 0.0001, "wave": 0.0001},
                "latency_ms": 8.65,
                "sequence_length": 30,
                "features_per_frame": 132
            }
        }
    )

class BatchPredictRequest(BaseModel):
    """Request payload for batched predictions."""
    sequences: List[List[List[float]]] = Field(
        ..., 
        description="List of sequences, each of shape (30, 132).",
        json_schema_extra={"example": [[[0.0] * 132] * 30] * 2}
    )
    return_probabilities: bool = Field(
        default=False,
        description="Whether to return full class probabilities for each sequence."
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sequences": [[[0.0] * 132] * 30] * 2,
                "return_probabilities": False
            }
        }
    )

class BatchPredictResponse(BaseModel):
    """Response payload for batched predictions."""
    results: List[PredictResponse] = Field(..., description="List of prediction results.")
    batch_latency_ms: float = Field(..., description="Total batch inference latency in ms.")


# ---------------------------------------------------------------------
# App Lifespan and Middleware
# ---------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to load the model on startup and 
    clean up on shutdown.
    """
    model_path = os.getenv("MODEL_PATH", "models/best_lstm.pt")
    
    print("\n--- Starting Pose Action Recognition API ---")
    try:
        classifier = ActionClassifier(model_path=model_path, device='cpu')
        app.state.classifier = classifier
        print("API started — model loaded successfully.")
    except Exception as e:
        print(f"FAILED to load model from {model_path}: {e}")
        app.state.classifier = None
        
    yield
    
    print("\nAPI shutting down.")


app = FastAPI(
    title="Pose Action Recognition API",
    description="REST API for serving LSTM/Transformer pose sequence classification.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log request method, path, status, and latency."""
    start_time = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start_time) * 1000
    print(f"{request.method} {request.url.path} | Status: {response.status_code} | Latency: {latency_ms:.2f}ms")
    return response


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------

@app.get("/")
async def root() -> Dict[str, Any]:
    """Returns basic API info and metadata."""
    classifier: ActionClassifier = getattr(app.state, "classifier", None)
    
    # Defaults in case model failed to load
    model_name = "Unknown"
    classes = []
    status_msg = "running" if classifier else "model_load_failed"
    
    if classifier:
        info = classifier.get_model_info()
        model_name = "ActionLSTM" if "lstm" in info['model_path'].lower() else "ActionTransformer"
        classes = list(info['label_map'].values())
        
    return {
        "name": "Pose Action Recognition API",
        "version": "1.0.0",
        "status": status_msg,
        "model": model_name,
        "classes": classes,
        "docs": "/docs"
    }


@app.get("/health")
async def health_check() -> JSONResponse:
    """Runs a health check with a dummy inference to ensure model readiness."""
    classifier: ActionClassifier = getattr(app.state, "classifier", None)
    
    if not classifier:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "detail": "Model not loaded"}
        )
        
    try:
        # Dummy inference
        dummy_seq = np.zeros((30, 132), dtype=np.float32)
        result = classifier.predict(dummy_seq)
        
        return JSONResponse(content={
            "status": "healthy",
            "model_loaded": True,
            "device": classifier.device.type,
            "val_accuracy": classifier.val_accuracy,
            "latency_ms": round(result['latency_ms'], 2)
        })
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "detail": f"Inference failed: {str(e)}"}
        )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predicts the action class from a single keypoint sequence."""
    classifier: ActionClassifier = getattr(app.state, "classifier", None)
    if not classifier:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    # Shape validation
    seq = request.sequence
    if len(seq) != 30 or any(len(frame) != 132 for frame in seq):
        raise HTTPException(
            status_code=422, 
            detail="Sequence shape must be exactly (30, 132). Please pad or truncate before sending."
        )
        
    try:
        np_seq = np.array(seq, dtype=np.float32)
        result = classifier.predict(np_seq)
        
        response = PredictResponse(
            action=result['action'],
            confidence=result['confidence'],
            probabilities=result['probabilities'] if request.return_probabilities else None,
            latency_ms=result['latency_ms'],
            sequence_length=30,
            features_per_frame=132
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "Internal inference error", "detail": str(e)})


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """Predicts the action classes for a batch of keypoint sequences."""
    classifier: ActionClassifier = getattr(app.state, "classifier", None)
    if not classifier:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    np_sequences = []
    for i, seq in enumerate(request.sequences):
        if len(seq) != 30 or any(len(frame) != 132 for frame in seq):
            raise HTTPException(
                status_code=422, 
                detail=f"Sequence at index {i} does not have shape (30, 132)."
            )
        np_sequences.append(np.array(seq, dtype=np.float32))
        
    start_time = time.perf_counter()
    
    try:
        results = classifier.predict_batch(np_sequences)
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        
        response_list = []
        for res in results:
            response_list.append(PredictResponse(
                action=res['action'],
                confidence=res['confidence'],
                probabilities=res['probabilities'] if request.return_probabilities else None,
                latency_ms=res['latency_ms'],
                sequence_length=30,
                features_per_frame=132
            ))
            
        return BatchPredictResponse(
            results=response_list,
            batch_latency_ms=total_latency_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "Internal batch inference error", "detail": str(e)})


@app.get("/model/info")
async def model_info() -> Dict[str, Any]:
    """Returns metadata and hyperparameter information for the loaded model."""
    classifier: ActionClassifier = getattr(app.state, "classifier", None)
    if not classifier:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    return classifier.get_model_info()


@app.get("/classes")
async def classes() -> Dict[str, Any]:
    """Returns the supported action classes and their label mappings."""
    classifier: ActionClassifier = getattr(app.state, "classifier", None)
    if not classifier:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    # We swap keys and values since label_map originally is {index: name}
    label_map = {name: idx for idx, name in classifier.reverse_label_map.items()}
    class_names = list(label_map.keys())
    
    return {
        "classes": class_names,
        "label_map": label_map,
        "num_classes": classifier.num_classes
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
