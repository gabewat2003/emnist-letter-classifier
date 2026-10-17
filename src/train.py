"""
train.py - Training script for the EMNIST CNN classifier.
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.data import get_data_loaders
from src.model import CNN

def train_model(epochs=10, batch_size=128, lr=1e-3, seed=42, device_str="cuda"):
    # Set reproducibility seed
    torch.manual_seed(seed)
    if torch.cuda.is_available() and "cuda" in device_str:
        torch.cuda.manual_seed_all(seed)
        device = torch.device(device_str)
    else:
      device = torch.device("cpu")
      
    print(f"Using device: {device}")
    
    # Create necessary directories
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_loader, test_loader, train_dataset, test_dataset = get_data_loaders(batch_size=batch_size)
    
    # Check label range gotcha (EMNIST letters often run 1-26)
    sample_labels = train_dataset.targets
    print(f"Dataset targets range before adjustment: min={sample_labels.min()}, max={sample_labels.max()}")
    
    model = CNN(num_classes=26).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # EMNIST letters split labels are 1-26; shift to 0-25 if needed
            if y.min() >= 1 and y.max() <= 26:
                y = y - 1
                
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total_train += y.size(0)
            correct_train += predicted.eq(y).sum().item()
            
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train
        
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                if y.min() >= 1 and y.max() <= 26:
                    y = y - 1
                    
                outputs = model(x)
                loss = criterion(outputs, y)
                
                test_loss += loss.item() * x.size(0)
                _, predicted = outputs.max(1)
                total_test += y.size(0)
                correct_test += predicted.eq(y).sum().item()
                
        test_loss = test_loss / total_test
        test_acc = correct_test / total_test
        
        print(f"Epoch [{epoch}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")
              
        # Save best checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint_path = "checkpoints/best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, checkpoint_path)
            print(f"--> Saved new best model checkpoint to {checkpoint_path} (Acc: {test_acc*100:.2f}%)")
            
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds. Best Test Accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EMNIST Letter Classifier CNN")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed, device_str=args.device)
