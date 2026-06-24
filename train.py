import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from model.st_gcn import STGCN
from dataset import SkeletalDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train ST-GCN Model for Slip and Fall Detection")
    parser.add_argument("--data_dir", type=str, default="data_processed",
                        help="Path to processed dataset directory.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and validation.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0001,
                        help="Weight decay.")
    parser.add_argument("--model_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--log_dir", type=str, default="runs",
                        help="TensorBoard logs directory.")
    parser.add_argument("--early_stopping", type=int, default=10,
                        help="Patience (epochs) for early stopping.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint file to resume training.")
    parser.add_argument("--mixed_precision", action="store_true", default=True,
                        help="Use PyTorch automatic mixed precision (AMP) on CUDA.")
    parser.add_argument("--mock", action="store_true",
                        help="Generate mock data and train to verify pipeline operations.")
    return parser.parse_args()

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def generate_mock_data(data_dir):
    """
    Generates synthetic skeletal keypoints for pipeline verification.
    Shapes mimic processed video sequences: (N, C, T, V, M)
    where N=Samples, C=3 (x,y,conf), T=50, V=17 (COCO), M=1
    """
    os.makedirs(data_dir, exist_ok=True)
    print("Generating mock dataset splits...")
    
    # Train: 120 samples, Val: 30 samples, Test: 30 samples
    x_train = np.random.randn(120, 3, 50, 17, 1).astype(np.float32)
    # Add a mock pattern: falls (label 1) have larger coordinate changes in later frames
    y_train = np.random.choice([0, 1], size=120).astype(np.int64)
    for i in range(len(y_train)):
        if y_train[i] == 1:
            # Fall posture has lower y values/larger offsets in lower joints (knees, ankles) towards the end
            x_train[i, 1, 30:, 11:] += 1.5 
            
    x_val = np.random.randn(30, 3, 50, 17, 1).astype(np.float32)
    y_val = np.random.choice([0, 1], size=30).astype(np.int64)
    
    x_test = np.random.randn(30, 3, 50, 17, 1).astype(np.float32)
    y_test = np.random.choice([0, 1], size=30).astype(np.int64)
    
    np.save(os.path.join(data_dir, "x_train.npy"), x_train)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "x_val.npy"), x_val)
    np.save(os.path.join(data_dir, "y_val.npy"), y_val)
    np.save(os.path.join(data_dir, "x_test.npy"), x_test)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)
    print(f"Mock dataset created successfully in {data_dir}.")

def train_epoch(model, loader, criterion, optimizer, scaler, device, mixed_precision):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # Mixed Precision execution
        with torch.cuda.amp.autocast(enabled=mixed_precision):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        if mixed_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    args = parse_args()
    
    if args.mock:
        generate_mock_data(args.data_dir)
        
    # Check if files exist
    required_files = ["x_train.npy", "y_train.npy", "x_val.npy", "y_val.npy"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(args.data_dir, f))]
    if missing:
        print(f"Error: Missing processed split files: {missing}")
        print("Please run `preprocess.py` first, or run with `--mock` to verify the code using synthetic data.")
        return
        
    # Set hardware device
    if not torch.cuda.is_available():
        raise RuntimeError("GPU (CUDA) is not reachable or not available, but forced GPU execution was requested!")
    device = torch.device("cuda")
    print(f"Training on hardware device: {device}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
        
    # Create datasets and dataloaders
    train_dataset = SkeletalDataset(
        x_path=os.path.join(args.data_dir, "x_train.npy"),
        y_path=os.path.join(args.data_dir, "y_train.npy"),
        augment=True
    )
    val_dataset = SkeletalDataset(
        x_path=os.path.join(args.data_dir, "x_val.npy"),
        y_path=os.path.join(args.data_dir, "y_val.npy"),
        augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Instantiate ST-GCN Model
    model = STGCN(in_channels=3, num_classes=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision and device.type == 'cuda'))
    early_stop = EarlyStopping(patience=args.early_stopping)
    writer = SummaryWriter(args.log_dir)
    
    os.makedirs(args.model_dir, exist_ok=True)
    start_epoch = 1
    best_val_loss = float('inf')
    
    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Resuming training from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scaler_state_dict' in checkpoint and scaler is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Resumed from epoch {checkpoint['epoch']} with best val loss of {best_val_loss:.4f}")
        else:
            print(f"Error: No checkpoint found at {args.resume}")
            return
            
    print(f"Starting training pipeline for {args.epochs} epochs...")
    
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, args.mixed_precision and device.type == 'cuda')
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
        
        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:6.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:6.2f}%")
              
        # Save checkpoints
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'best_val_loss': best_val_loss
            }, os.path.join(args.model_dir, "best_model.pt"))
            print(f"  --> Saved new best model weights (Val Loss: {val_loss:.4f})")
            
        # Save latest checkpoint for resume capability
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'best_val_loss': best_val_loss
        }, os.path.join(args.model_dir, "latest_checkpoint.pt"))
        
        # Check early stopping
        early_stop(val_loss)
        if early_stop.early_stop:
            print(f"Early stopping triggered at epoch {epoch}. Validation loss hasn't improved.")
            break
            
    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    main()
