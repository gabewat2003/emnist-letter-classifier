"""
evaluate.py - Evaluation script for computing metrics, confusion matrix, and error galleries.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path

from src.data import get_data_loaders
from src.model import CNN

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def evaluate_model(checkpoint_path="checkpoints/best.pt"):
    device = get_device()
    print(f"Evaluating using device: {device}")
    
    # Disable pin_memory if on MPS to avoid warnings
    pin_memory = device.type == "cuda"
    _, test_loader, _, _ = get_data_loaders()
    
    model = CNN(num_classes=26).to(device)
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please train the model first.")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x_dev = x.to(device)
            y_dev = y.to(device)
            if y_dev.min() >= 1 and y_dev.max() <= 26:
                y_dev = y_dev - 1
                
            outputs = model(x_dev)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y_dev.cpu().numpy())
                    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    accuracy = np.mean(all_preds == all_targets)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Confusion Matrix (rows: true, columns: predicted)
    cm = confusion_matrix(all_targets, all_preds, labels=range(26))
    
    # Find most confused letter pairs (excluding self)
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    confusions = []
    classes = [chr(ord('A') + i) for i in range(26)]
    
    for i in range(26):
        for j in range(26):
            if i != j and cm_no_diag[i, j] > 0:
                confusions.append((cm_no_diag[i, j], i, j))
                
    confusions.sort(key=lambda x: x[0], reverse=True)
    
    print("\nTop 3 Most-Confused Letter Pairs (True -> Predicted):")
    for count, true_cls, pred_cls in confusions[:3]:
        print(f"  True '{classes[true_cls]}' misclassified as '{classes[pred_cls]}': {count} times")
        
    # Plot Confusion Matrix cleanly
    plt.figure(figsize=(11, 9))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("EMNIST Letters Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(26)
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    Path("results").mkdir(parents=True, exist_ok=True)
    plt.savefig("results/confusion_matrix.png", dpi=300)
    print("Saved confusion matrix plot to results/confusion_matrix.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EMNIST Letter Classifier CNN")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to model checkpoint")
    args = parser.parse_args()
    evaluate_model(checkpoint_path=args.checkpoint)
