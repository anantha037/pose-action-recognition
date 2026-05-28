"""
app/inference.py

Inference engine for the Action Recognition model. Loads a trained PyTorch
model checkpoint and provides an easy-to-use interface for sequence prediction.
"""

import time
import warnings
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

import sys
import os
# Add parent directory to path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model architectures
from app.model import ActionLSTM, ActionTransformer, count_parameters


class ActionClassifier:
    """
    Inference wrapper for the trained pose action recognition model.
    Handles data preprocessing, model execution, and probability formatting.
    """
    
    def __init__(self, model_path: str = 'models/best_lstm.pt', device: str = 'cpu'):
        """
        Initializes the ActionClassifier.
        
        Args:
            model_path (str): Path to the PyTorch checkpoint (.pt file).
            device (str): Compute device ('cpu' or 'cuda').
        """
        self.model_path = model_path
        self.device = torch.device(device)
        self.sequence_length = 30  # Default expected sequence length
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Extract metadata
        self.epoch = checkpoint.get('epoch', 0)
        self.val_accuracy = checkpoint.get('val_accuracy', 0.0)
        
        # Reconstruct label map
        label_map = checkpoint.get('label_map', {})
        self.reverse_label_map = {idx: name for name, idx in label_map.items()}
        self.num_classes = len(self.reverse_label_map)
        
        # Reconstruct config
        config = checkpoint.get('config', {})
        model_type = config.get('model_type', 'lstm').lower()
        
        # Extract normalization stats
        if 'mean' in checkpoint and 'std' in checkpoint:
            self.mean = checkpoint['mean']
            self.std = checkpoint['std']
        else:
            warnings.warn("Mean and std not found in checkpoint! Inference will run with fallback normalization (zeros/ones), which may severely degrade accuracy.")
            self.mean = np.zeros(132, dtype=np.float32)
            self.std = np.ones(132, dtype=np.float32)
            
        # Instantiate model based on config
        if model_type == 'lstm':
            self.model = ActionLSTM(
                input_size=132,
                hidden_size=config.get('hidden_size', 256),
                num_layers=config.get('num_layers', 2),
                num_classes=self.num_classes,
                dropout=config.get('dropout', 0.3)
            )
        elif model_type == 'transformer':
            self.model = ActionTransformer(
                input_size=132,
                d_model=config.get('d_model', 128),
                nhead=config.get('nhead', 4),
                num_layers=config.get('num_layers', 2),
                num_classes=self.num_classes,
                dropout=config.get('dropout', 0.3)
            )
        else:
            raise ValueError(f"Unknown model_type in checkpoint config: {model_type}")
            
        # Load weights and set to eval mode
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Pre-compute parameter count
        self.total_params, _ = count_parameters(self.model)
        
        print(f"ActionClassifier loaded | Model: {model_type.upper()} | Classes: {self.num_classes} | "
              f"Device: {self.device.type} | Val accuracy: {self.val_accuracy*100:.1f}%")

    def preprocess(self, sequence: np.ndarray) -> torch.Tensor:
        """
        Preprocesses a raw keypoint sequence to match training format.
        
        Args:
            sequence (np.ndarray): Raw sequence array of shape (frames, 132).
            
        Returns:
            torch.Tensor: Preprocessed tensor of shape (1, sequence_length, 132).
        """
        seq = sequence.copy()
        
        # Handle shape mismatches
        if len(seq) < self.sequence_length:
            pad_amount = self.sequence_length - len(seq)
            seq = np.pad(seq, ((0, pad_amount), (0, 0)), mode='constant', constant_values=0)
        elif len(seq) > self.sequence_length:
            start = (len(seq) - self.sequence_length) // 2
            seq = seq[start : start + self.sequence_length]
            
        # Z-score normalization using saved training statistics
        seq = (seq - self.mean) / (self.std + 1e-8)
        
        # Convert to tensor, add batch dimension, move to device
        tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def predict(self, sequence: np.ndarray) -> Dict[str, Any]:
        """
        Runs a forward pass on a single sequence and returns class probabilities.
        
        Args:
            sequence (np.ndarray): Raw keypoint sequence of shape (frames, 132).
            
        Returns:
            Dict[str, Any]: Dictionary containing prediction results and latency.
        """
        start_time = time.perf_counter()
        
        # Preprocess
        tensor = self.preprocess(sequence)
        
        # Forward pass
        logits = self.model(tensor)
        
        # Probabilities
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Format output
        pred_idx = int(np.argmax(probs))
        pred_action = self.reverse_label_map[pred_idx]
        confidence = float(probs[pred_idx])
        
        prob_dict = {self.reverse_label_map[i]: float(p) for i, p in enumerate(probs)}
        
        return {
            'action': pred_action,
            'confidence': confidence,
            'probabilities': prob_dict,
            'latency_ms': latency_ms
        }

    @torch.no_grad()
    def predict_batch(self, sequences: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Runs predictions on a batch of sequences efficiently.
        
        Args:
            sequences (List[np.ndarray]): List of raw keypoint sequences.
            
        Returns:
            List[Dict[str, Any]]: List of prediction result dictionaries.
        """
        start_time = time.perf_counter()
        
        # Preprocess all sequences
        tensors = []
        for seq in sequences:
            # We squeeze to remove the batch dim from preprocess, then we will stack
            tensors.append(self.preprocess(seq).squeeze(0))
            
        # Stack into a single batch tensor: (batch_size, sequence_length, 132)
        batch_tensor = torch.stack(tensors).to(self.device)
        
        # Forward pass
        logits = self.model(batch_tensor)
        
        # Probabilities
        probs_batch = F.softmax(logits, dim=1).cpu().numpy()
        
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        per_sample_latency = total_latency_ms / len(sequences)
        
        # Format output
        results = []
        for probs in probs_batch:
            pred_idx = int(np.argmax(probs))
            pred_action = self.reverse_label_map[pred_idx]
            confidence = float(probs[pred_idx])
            
            prob_dict = {self.reverse_label_map[j]: float(p) for j, p in enumerate(probs)}
            
            results.append({
                'action': pred_action,
                'confidence': confidence,
                'probabilities': prob_dict,
                'latency_ms': per_sample_latency
            })
            
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns metadata and configuration details about the loaded model.
        
        Returns:
            Dict[str, Any]: Model information dictionary.
        """
        return {
            'model_path': self.model_path,
            'num_classes': self.num_classes,
            'label_map': self.reverse_label_map,
            'device': self.device.type,
            'val_accuracy': self.val_accuracy,
            'epoch': self.epoch,
            'parameter_count': self.total_params
        }


if __name__ == "__main__":
    import os
    
    # Create a dummy model checkpoint for testing if one doesn't exist
    test_model_path = 'models/best_lstm.pt'
    if not os.path.exists(test_model_path):
        print(f"Warning: '{test_model_path}' not found. Generating a dummy checkpoint to run the smoke test...\n")
        
        os.makedirs('models', exist_ok=True)
        dummy_model = ActionLSTM()
        torch.save({
            'model_state_dict': dummy_model.state_dict(),
            'optimizer_state_dict': {},
            'epoch': 1,
            'val_accuracy': 0.85,
            'label_map': {'idle': 0, 'jumping_jack': 1, 'push_up': 2, 'squat': 3, 'wave': 4},
            'config': {'model_type': 'lstm', 'hidden_size': 256, 'num_layers': 2, 'dropout': 0.3},
            'mean': np.zeros(132, dtype=np.float32),
            'std': np.ones(132, dtype=np.float32)
        }, test_model_path)
    
    # 1. Instantiate ActionClassifier
    classifier = ActionClassifier(model_path=test_model_path, device='cpu')
    
    # 2. Print model info
    info = classifier.get_model_info()
    print("\n--- Model Info ---")
    for k, v in info.items():
        print(f"{k}: {v}")
        
    # 3. Create dummy sequence and predict
    dummy_seq = np.zeros((30, 132), dtype=np.float32)
    result = classifier.predict(dummy_seq)
    
    print("\n--- Single Prediction Result ---")
    for k, v in result.items():
        if k == 'probabilities':
            print(f"{k}:")
            for action, p in v.items():
                print(f"  {action}: {p:.4f}")
        else:
            print(f"{k}: {v}")
            
    # 4. Measure average latency over 100 predictions
    print("\nMeasuring average latency over 100 predictions...")
    latencies = []
    for _ in range(100):
        # Generate random inputs to prevent caching optimizations from skewing results
        random_seq = np.random.randn(30, 132).astype(np.float32)
        res = classifier.predict(random_seq)
        latencies.append(res['latency_ms'])
        
    avg_latency = sum(latencies) / len(latencies)
    print(f"Average latency per frame sequence: {avg_latency:.2f} ms")
