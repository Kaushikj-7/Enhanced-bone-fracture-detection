import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import get_dataloaders
from model import HybridCNNViT
import time
import copy
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import numpy as np


def train_model(
    model, dataloaders, criterion, optimizer, device, num_epochs=20, patience=5
):
    since = time.time()

    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    early_stopping_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_auc": [],
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Use lists to collect all preds and labels for epoch metrics
            epoch_preds = []
            epoch_labels = []

            # Check if dataloader is valid
            if (
                phase not in dataloaders
                or not hasattr(dataloaders[phase], "dataset")
                or len(dataloaders[phase].dataset) == 0
            ):
                continue

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)  # Logits
                    loss = criterion(outputs, labels)

                    # Sigmoid for predictions
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()

                    # Backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)

                # Collect for metrics
                epoch_labels.extend(labels.cpu().detach().numpy())
                epoch_preds.extend(probs.cpu().detach().numpy())

            dataset_size = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / dataset_size

            # Calculate Metrics
            epoch_labels = np.array(epoch_labels)
            epoch_probs = np.array(epoch_preds)
            epoch_binary_preds = (epoch_probs > 0.5).astype(int)

            epoch_acc = accuracy_score(epoch_labels, epoch_binary_preds)

            if phase == "val":
                epoch_precision = precision_score(
                    epoch_labels, epoch_binary_preds, zero_division=0
                )
                epoch_recall = recall_score(
                    epoch_labels, epoch_binary_preds, zero_division=0
                )
                epoch_f1 = f1_score(epoch_labels, epoch_binary_preds, zero_division=0)

                try:
                    epoch_auc = 0.5
                    if len(np.unique(epoch_labels)) > 1:
                        fpr, tpr, _ = roc_curve(epoch_labels, epoch_probs)
                        epoch_auc = auc(fpr, tpr)
                except:
                    epoch_auc = 0.5

                print(
                    f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Prec: {epoch_precision:.4f} Rec: {epoch_recall:.4f} F1: {epoch_f1:.4f} AUC: {epoch_auc:.4f}"
                )

                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc)
                history["val_precision"].append(epoch_precision)
                history["val_recall"].append(epoch_recall)
                history["val_f1"].append(epoch_f1)
                history["val_auc"].append(epoch_auc)

                # Deep copy the model
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
            else:
                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc)

        if early_stopping_counter >= patience:
            print("Early stopping triggered")
            break

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Val Loss: {best_loss:.4f}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            all_preds.extend(probs.cpu().detach().numpy())
            all_labels.extend(labels.cpu().detach().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_preds)
    binary_preds = (all_probs > 0.5).astype(int)

    # Calculate metrics
    acc = accuracy_score(all_labels, binary_preds)
    prec = precision_score(all_labels, binary_preds, zero_division=0)
    rec = recall_score(all_labels, binary_preds, zero_division=0)
    f1 = f1_score(all_labels, binary_preds, zero_division=0)

    auc_score = 0.5
    fpr, tpr = [0, 1], [0, 1]

    try:
        if len(np.unique(all_labels)) > 1:
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            auc_score = auc(fpr, tpr)
    except:
        pass

    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc_roc": auc_score,
        "fpr": fpr,
        "tpr": tpr,
    }

    print(f"Evaluation Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  AUC-ROC: {auc_score:.4f}")

    return results
