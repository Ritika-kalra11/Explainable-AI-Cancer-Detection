"""
Step 3: Baseline ResNet-50 Training on Leakage-Free HAM10000 Dataset
Binary Classification:
- Class 0: Benign (nv, bkl, vasc, df)
- Class 1: Malignant/Premalignant (mel, bcc, akiec)

Model: Pretrained ResNet-50 (ImageNet weights)
Split: Lesion-level leakage-free split (train.csv, validation.csv, test.csv)
Loss: Standard Cross-Entropy Loss (unweighted)
Metrics: Accuracy, Malignant Recall/Sensitivity, Specificity, Precision, F1-Score, ROC-AUC
"""

from pathlib import Path
import time
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# 1. Project Paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "HAM10000"
SPLIT_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 2. PyTorch Dataset
class HAM10000BinaryDataset(Dataset):
    def __init__(self, df, transform=None, fallback_images_dir=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.fallback_images_dir = fallback_images_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(str(row["image_path"]))

        if not img_path.is_file() and self.fallback_images_dir:
            alt_path = self.fallback_images_dir / img_path.name
            if alt_path.is_file():
                img_path = alt_path

        image = Image.open(img_path).convert("RGB")
        label = int(row["binary_label"])

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms():
    # Normalization using standard ImageNet mean and std
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    return train_transform, eval_transform

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_targets.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    total_samples = len(all_targets)
    avg_loss = total_loss / total_samples

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    malignant_recall = recall_score(all_targets, all_preds, pos_label=1, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(all_targets, all_preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_targets, all_preds, pos_label=1, zero_division=0)

    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.5

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "malignant_recall": malignant_recall,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }

def train_baseline(epochs=3, batch_size=64, lr=2e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(8)

    print("======================================================================")
    print("STEP 3: BASELINE RESNET-50 TRAINING (LEAKAGE-FREE SPLIT)")
    print("======================================================================")
    print(f"Device:                 {device} (Hardware Threads: 8)")
    print(f"Epochs:                 {epochs}")
    print(f"Batch Size:             {batch_size}")
    print(f"Initial Learning Rate:  {lr}")
    print("Loss Function:          Standard Cross-Entropy (Unweighted Baseline)")
    print("----------------------------------------------------------------------")

    # Load splits
    train_df = pd.read_csv(SPLIT_DIR / "train.csv")
    val_df = pd.read_csv(SPLIT_DIR / "validation.csv")
    test_df = pd.read_csv(SPLIT_DIR / "test.csv")

    fallback_img_dir = DATA_DIR / "images"

    train_tf, eval_tf = get_transforms()

    train_dataset = HAM10000BinaryDataset(train_df, transform=train_tf, fallback_images_dir=fallback_img_dir)
    val_dataset = HAM10000BinaryDataset(val_df, transform=eval_tf, fallback_images_dir=fallback_img_dir)
    test_dataset = HAM10000BinaryDataset(test_df, transform=eval_tf, fallback_images_dir=fallback_img_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Dataset Counts:")
    print(f"  • Training:   {len(train_dataset)} images ({len(train_loader)} batches)")
    print(f"  • Validation: {len(val_dataset)} images ({len(val_loader)} batches)")
    print(f"  • Testing:    {len(test_dataset)} images ({len(test_loader)} batches) [Untouched during training]")
    print("----------------------------------------------------------------------")

    # Initialize Pretrained ResNet-50
    print("Loading pretrained ResNet-50 backbone...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze earlier convolutional blocks (conv1, layer1, layer2, layer3) for stable feature extraction on CPU
    for name, param in model.named_parameters():
        if not (name.startswith("layer4") or name.startswith("fc")):
            param.requires_grad = False

    # Replace classifier head for 2 classes (Benign: 0, Malignant/Premalignant: 1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 2)
    )

    model = model.to(device)

    # Standard Cross-Entropy Loss
    criterion = nn.CrossEntropyLoss()

    # Differential learning rates for fine-tuning
    optimizer = torch.optim.AdamW([
        {"params": model.layer4.parameters(), "lr": lr * 0.5},
        {"params": model.fc.parameters(), "lr": lr}
    ], weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_epoch = 0
    checkpoint_path = MODELS_DIR / "baseline_resnet50_best.pth"

    training_start_time = time.time()
    history = []

    print("\n--- Starting Training & Validation ---")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size_cur = images.size(0)
            running_loss += loss.item() * batch_size_cur
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += batch_size_cur

            # Periodic batch progress
            if batch_idx % 25 == 0 or batch_idx == len(train_loader):
                cur_loss = running_loss / total_train
                cur_acc = (correct_train / total_train) * 100
                elapsed = time.time() - epoch_start
                img_per_sec = total_train / elapsed if elapsed > 0 else 0
                print(f"  [Epoch {epoch}/{epochs}] Batch {batch_idx:>3}/{len(train_loader)} | Train Loss: {cur_loss:.4f} | Train Acc: {cur_acc:>5.2f}% | Speed: {img_per_sec:>4.1f} img/s")

        scheduler.step()
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        # Validation at epoch end
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_accuracy": round(val_metrics["accuracy"], 4),
            "val_malignant_recall": round(val_metrics["malignant_recall"], 4),
            "val_specificity": round(val_metrics["specificity"], 4),
            "val_f1": round(val_metrics["f1"], 4),
            "val_roc_auc": round(val_metrics["roc_auc"], 4),
            "epoch_duration_sec": round(epoch_time, 1)
        }
        history.append(epoch_record)

        print(f"\n  >> End of Epoch {epoch} ({epoch_time:.1f}s):")
        print(f"     Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}% | Val AUC: {val_metrics['roc_auc']:.4f} | Malignant Recall: {val_metrics['malignant_recall']*100:.2f}%\n")

        # Save Best Checkpoint based on Validation Loss
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
            print(f"     [NEW BEST MODEL] Saved checkpoint to {checkpoint_path.name} (Val Loss: {best_val_loss:.4f})")

    total_training_duration = time.time() - training_start_time
    print(f"\nTraining completed in {total_training_duration/60:.2f} minutes.")
    print(f"Selected best model checkpoint from Epoch {best_epoch} (Val Loss: {best_val_loss:.4f}).")

    # Final Untouched Test Set Evaluation
    print("\n======================================================================")
    print("STEP 3 (FINAL): UNTOUCHED TEST SET EVALUATION")
    print("======================================================================")
    print(f"Loading best checkpoint from: {checkpoint_path.name}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_metrics = evaluate_model(model, test_loader, criterion, device)

    # Format Results
    cm = test_metrics["confusion_matrix"]
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]

    print("\n---------------- FINAL TEST PERFORMANCE METRICS ----------------")
    print(f"  • Test Accuracy:                 {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  • Malignant Recall / Sensitivity: {test_metrics['malignant_recall'] * 100:.2f}% ({tp}/{tp + fn} cancer cases detected)")
    print(f"  • Benign Specificity:            {test_metrics['specificity'] * 100:.2f}% ({tn}/{tn + fp} benign cases correctly identified)")
    print(f"  • Precision (Malignant):         {test_metrics['precision'] * 100:.2f}%")
    print(f"  • F1-Score:                      {test_metrics['f1']:.4f}")
    print(f"  • ROC-AUC Score:                 {test_metrics['roc_auc']:.4f}")
    print("-----------------------------------------------------------------")
    print(f"Confusion Matrix:")
    print(f"                 Predicted Benign (0)  |  Predicted Malignant/Premalignant (1)")
    print(f"Actual Benign:          {tn:<21} |  {fp:<36}")
    print(f"Actual Malignant:       {fn:<21} |  {tp:<36}")
    print("-----------------------------------------------------------------")

    # Assemble Full Report
    final_report = {
        "model_architecture": "ResNet-50 (Pretrained ImageNet, layer4+fc fine-tuned)",
        "training_strategy": "Baseline Standard Cross-Entropy (Unweighted)",
        "split_method": "Lesion-level leakage-free split (GroupShuffleSplit)",
        "best_epoch": best_epoch,
        "total_training_minutes": round(total_training_duration / 60, 2),
        "history": history,
        "test_metrics": {
            "accuracy": round(test_metrics["accuracy"], 4),
            "malignant_recall": round(test_metrics["malignant_recall"], 4),
            "specificity": round(test_metrics["specificity"], 4),
            "precision": round(test_metrics["precision"], 4),
            "f1_score": round(test_metrics["f1"], 4),
            "roc_auc": round(test_metrics["roc_auc"], 4),
            "confusion_matrix": cm
        }
    }

    report_json_path = REPORTS_DIR / "baseline_resnet50_metrics.json"
    report_txt_path = REPORTS_DIR / "baseline_resnet50_metrics.txt"

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4)

    txt_lines = [
        "======================================================================",
        "HAM10000 BASELINE RESNET-50 EVALUATION REPORT",
        "======================================================================",
        f"Architecture:      ResNet-50 (Pretrained ImageNet)",
        f"Splitting:         Lesion-Level Leakage-Free Split (0% overlap)",
        f"Loss:              Standard Cross-Entropy Loss (Unweighted)",
        f"Best Epoch:        {best_epoch} of {epochs}",
        f"Training Time:     {total_training_duration/60:.2f} minutes\n",
        "FINAL TEST PERFORMANCE (Untouched test.csv, 1,494 images):",
        "----------------------------------------------------------------------",
        f"Test Accuracy:                  {test_metrics['accuracy'] * 100:.2f}%",
        f"Malignant Recall (Sensitivity):  {test_metrics['malignant_recall'] * 100:.2f}%",
        f"Specificity (Benign Recall):    {test_metrics['specificity'] * 100:.2f}%",
        f"Precision (Malignant):          {test_metrics['precision'] * 100:.2f}%",
        f"F1-Score:                       {test_metrics['f1']:.4f}",
        f"ROC-AUC Score:                  {test_metrics['roc_auc']:.4f}\n",
        "CONFUSION MATRIX:",
        f"  • True Negatives (TN) [Benign Correct]:              {tn}",
        f"  • False Positives (FP) [Benign Called Cancer]:        {fp}",
        f"  • False Negatives (FN) [Cancer Missed]:               {fn}",
        f"  • True Positives (TP) [Cancer Detected]:              {tp}\n",
        "EPOCH TRAINING HISTORY:",
        "----------------------------------------------------------------------"
    ]
    for h in history:
        txt_lines.append(f"Epoch {h['epoch']}: Train Loss={h['train_loss']:.4f}, Val Loss={h['val_loss']:.4f}, Val Acc={h['val_accuracy']*100:.2f}%, Val Recall={h['val_malignant_recall']*100:.2f}%, Val AUC={h['val_roc_auc']:.4f}")

    txt_lines.append("======================================================================")

    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines) + "\n")

    print(f"\nModel checkpoint saved: {checkpoint_path}")
    print(f"JSON metrics saved:     {report_json_path}")
    print(f"Text report saved:      {report_txt_path}")
    print("======================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    train_baseline(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
