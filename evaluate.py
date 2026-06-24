import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
)

from model.st_gcn import STGCN
from dataset import SkeletalDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained ST-GCN Model")
    parser.add_argument("--data_dir", type=str, default="data_processed",
                        help="Path to processed dataset directory.")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt",
                        help="Path to trained model weights checkpoint.")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation report and plots.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if files exist
    x_test_path = os.path.join(args.data_dir, "x_test.npy")
    y_test_path = os.path.join(args.data_dir, "y_test.npy")
    if not os.path.exists(x_test_path) or not os.path.exists(y_test_path):
        print(f"Error: Missing test split files: {x_test_path} or {y_test_path}")
        print("Please run `preprocess.py` (or run training with `--mock`) first to generate data.")
        return
        
    if not os.path.exists(args.model_path):
        print(f"Error: Trained model weights not found at: {args.model_path}")
        print("Please run `train.py` to train the model first.")
        return
        
    if not torch.cuda.is_available():
        raise RuntimeError("GPU (CUDA) is not reachable or not available, but forced GPU execution was requested for evaluation!")
    device = torch.device("cuda")
    print(f"Evaluating model on device: {device}")
    
    # Load test dataset
    test_dataset = SkeletalDataset(x_path=x_test_path, y_path=y_test_path, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Instantiate and load model weights
    model = STGCN(in_channels=3, num_classes=2).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model weights loaded successfully.")
    
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Convert logits to probabilities
            probs = torch.softmax(outputs, dim=1)
            
            # Predict labels
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            # Probability of the positive class (Fall, class 1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_targets.extend(targets.numpy())
            
    # Convert lists to NumPy arrays
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Calculate classification metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        # Avoid crashing if test set contains only one class
        roc_auc = float('nan')
        
    print("\n" + "="*40)
    print("           EVALUATION SUMMARY")
    print("="*40)
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("="*40)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate and save Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Fall', 'Fall'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax)
    plt.title("Confusion Matrix - ST-GCN Fall Detection")
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Confusion Matrix plot to: {cm_path}")
    
    # Generate and save ROC Curve
    if not np.isnan(roc_auc):
        fpr, tpr, _ = roc_curve(all_targets, all_probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.6)
        roc_path = os.path.join(args.output_dir, "roc_curve.png")
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved ROC Curve plot to: {roc_path}")
        
    # Write text report
    report_path = os.path.join(args.output_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write("ST-GCN Fall Detection System Evaluation Report\n")
        f.write("================================================\n\n")
        f.write(f"Accuracy:  {accuracy*100:.2f}%\n")
        f.write(f"Precision: {precision*100:.2f}%\n")
        f.write(f"Recall:    {recall*100:.2f}%\n")
        f.write(f"F1 Score:  {f1*100:.2f}%\n")
        f.write(f"ROC-AUC:   {roc_auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  Non-Fall vs Non-Fall (TN): {cm[0,0]}\n")
        f.write(f"  Non-Fall vs Fall     (FP): {cm[0,1]}\n")
        f.write(f"  Fall     vs Non-Fall (FN): {cm[1,0]}\n")
        f.write(f"  Fall     vs Fall     (TP): {cm[1,1]}\n")
    print(f"Saved text report to: {report_path}")

if __name__ == "__main__":
    main()
