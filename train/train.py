"""
train/train.py

Full PyTorch training loop for pose action recognition with MLflow tracking,
model checkpointing, and evaluation.
"""

import os
import sys
import random
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for environments like Colab
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

# Add parent directory to path so we can import from train and app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.dataset import get_data_loaders
from app.model import get_model


@dataclass
class TrainingConfig:
    """Dataclass holding all training hyperparameters and configuration."""
    data_dir: str = 'data/sequences'
    model_type: str = 'lstm'           # 'lstm' or 'transformer'
    num_epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    hidden_size: int = 256             # LSTM only
    num_layers: int = 2
    val_split: float = 0.2
    test_split: float = 0.1
    num_workers: int = 2
    seed: float = 42
    checkpoint_dir: str = 'models'
    mlflow_experiment: str = 'pose-action-recognition'
    early_stopping_patience: int = 10
    scheduler_patience: int = 5
    actions: List[str] = field(default_factory=lambda: ['idle', 'jumping_jack', 'push_up', 'squat', 'wave'])


def set_seed(seed: int) -> None:
    """
    Sets random seed for random, numpy, torch, torch.cuda for full reproducibility.
    
    Args:
        seed (int): The random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensures deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler
) -> Dict[str, float]:
    """
    Runs one full training epoch with mixed precision.
    
    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): DataLoader for training data.
        optimizer (optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run training on.
        scaler (torch.cuda.amp.GradScaler): GradScaler for mixed precision.
        
    Returns:
        Dict[str, float]: Dictionary containing average 'loss' and 'accuracy'.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with torch.amp.autocast(device_type=device.type):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = total_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return {'loss': epoch_loss, 'accuracy': epoch_acc}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, Any]:
    """
    Runs evaluation (no grad) on a given DataLoader.
    
    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): DataLoader for evaluation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run evaluation on.
        
    Returns:
        Dict[str, Any]: Dictionary containing 'loss', 'accuracy', 'predictions', and 'targets'.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.amp.autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())
            
    epoch_loss = total_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'predictions': all_preds,
        'targets': all_targets
    }


def plot_confusion_matrix(targets: List[int], predictions: List[int], class_names: List[str], save_path: str) -> str:
    """
    Plots a confusion matrix using seaborn heatmap and saves it as a PNG.
    
    Args:
        targets (List[int]): Ground truth labels.
        predictions (List[int]): Predicted labels.
        class_names (List[str]): List of class names for labels.
        save_path (str): Path to save the PNG image.
        
    Returns:
        str: The save_path.
    """
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str
) -> str:
    """
    Plots training and validation loss and accuracy curves and saves as a PNG.
    
    Args:
        train_losses (List[float]): List of training losses per epoch.
        val_losses (List[float]): List of validation losses per epoch.
        train_accs (List[float]): List of training accuracies per epoch.
        val_accs (List[float]): List of validation accuracies per epoch.
        save_path (str): Path to save the PNG image.
        
    Returns:
        str: The save_path.
    """
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


class EarlyStopping:
    """
    Early stopping to terminate training when a monitored metric stops improving.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        """
        Initializes EarlyStopping.
        
        Args:
            patience (int): Number of epochs to wait without improvement.
            min_delta (float): Minimum change to qualify as an improvement.
            mode (str): 'max' to monitor for maximum value (e.g., accuracy), 
                        'min' for minimum (e.g., loss).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_metric = -float('inf') if mode == 'max' else float('inf')
        self.counter = 0

    def step(self, metric: float) -> bool:
        """
        Checks if the metric improved. Returns True if training should stop.
        
        Args:
            metric (float): The current metric value.
            
        Returns:
            bool: True if early stopping is triggered, False otherwise.
        """
        if self.mode == 'max':
            improvement = metric - self.best_metric > self.min_delta
        else:
            improvement = self.best_metric - metric > self.min_delta
            
        if improvement:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience


def train(config: TrainingConfig) -> None:
    """
    Full training loop with MLflow tracking, checkpointing, and evaluation.
    
    Args:
        config (TrainingConfig): Configuration dataclass for training.
    """
    set_seed(int(config.seed))
    
    # MLflow setup
    mlflow.set_experiment(config.mlflow_experiment)
    
    with mlflow.start_run():
        mlflow.log_params(asdict(config))
        
        # Load Data
        loaders = get_data_loaders(
            data_dir=config.data_dir,
            actions=config.actions,
            batch_size=config.batch_size,
            val_split=config.val_split,
            test_split=config.test_split,
            num_workers=config.num_workers,
            seed=int(config.seed)
        )
        
        train_loader = loaders['train']
        val_loader = loaders['val']
        test_loader = loaders['test']
        label_map = loaders['label_map']
        class_weights = loaders['class_weights']
        
        if train_loader is None:
            print("Training failed to start because loaders were empty.")
            return
            
        # Create inverse label map for classification report plotting
        inv_label_map = {v: k for k, v in label_map.items()}
        class_names = [inv_label_map[i] for i in range(len(label_map))]
        
        # Initialize Model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_model(
            model_type=config.model_type,
            num_classes=len(config.actions),
            device=device,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout
        )
        
        # Setup Training specific objects
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=config.scheduler_patience, factor=0.5, verbose=True
        )
        scaler = torch.cuda.amp.GradScaler()
        early_stopping = EarlyStopping(patience=config.early_stopping_patience, mode='max')
        
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        best_model_path = os.path.join(config.checkpoint_dir, f"best_{config.model_type}.pt")
        
        # Tracking lists for plotting
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        best_val_acc = 0.0
        
        for epoch in range(1, config.num_epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
            val_metrics = evaluate(model, val_loader, criterion, device)
            
            # Step the scheduler using validation accuracy
            scheduler.step(val_metrics['accuracy'])
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record metrics
            train_losses.append(train_metrics['loss'])
            train_accs.append(train_metrics['accuracy'])
            val_losses.append(val_metrics['loss'])
            val_accs.append(val_metrics['accuracy'])
            
            # Log to MLflow
            mlflow.log_metrics({
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'lr': current_lr
            }, step=epoch)
            
            print(f"Epoch {epoch:02d}/{config.num_epochs:02d} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']*100:.1f}% | "
                  f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']*100:.1f}% | "
                  f"LR: {current_lr:.6f}")
            
            # Checkpoint
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_accuracy': best_val_acc,
                    'label_map': label_map,
                    'config': asdict(config),
                    'mean': loaders['mean'],
                    'std': loaders['std']
                }, best_model_path)
            
            # Early stopping check
            if early_stopping.step(val_metrics['accuracy']):
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
        # --- Post Training Evaluation ---
        print("\nEvaluating best model on test set...")
        # Load best model checkpoint for evaluation
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics = evaluate(model, test_loader, criterion, device)
        test_acc = test_metrics['accuracy']
        
        print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(test_metrics['targets'], test_metrics['predictions'], target_names=class_names))
        
        # Plotting
        os.makedirs("artifacts", exist_ok=True)
        cm_path = plot_confusion_matrix(
            targets=test_metrics['targets'],
            predictions=test_metrics['predictions'],
            class_names=class_names,
            save_path="artifacts/confusion_matrix.png"
        )
        
        curves_path = plot_training_curves(
            train_losses, val_losses, train_accs, val_accs,
            save_path="artifacts/training_curves.png"
        )
        
        # Log final artifacts and metrics to MLflow
        mlflow.log_metric('test_accuracy', test_acc)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(curves_path)
        mlflow.log_artifact(best_model_path)
        
        print(f"\nTraining complete. Best val accuracy: {best_val_acc*100:.1f}% | Test accuracy: {test_acc*100:.1f}%")


def main() -> None:
    """CLI entrypoint to parse arguments and start training."""
    parser = argparse.ArgumentParser(description="Train Pose Action Recognition Model")
    parser.add_argument('--model_type', type=str, help="Model type: 'lstm' or 'transformer'")
    parser.add_argument('--num_epochs', type=int, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, help="Batch size")
    parser.add_argument('--learning_rate', type=float, help="Learning rate")
    parser.add_argument('--hidden_size', type=int, help="Hidden size for LSTM model")
    parser.add_argument('--checkpoint_dir', type=str, help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    # Initialize default config and override with provided args
    config = TrainingConfig()
    for arg, value in vars(args).items():
        if value is not None:
            setattr(config, arg, value)
            
    train(config)


if __name__ == "__main__":
    main()
