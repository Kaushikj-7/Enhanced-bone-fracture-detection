import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from tqdm import tqdm
import time
import copy
from utils.metrics import compute_metrics, save_metrics


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Training")
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()

        # Automatic Mixed Precision (AMP)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Sigmoid for binary predictions
        probs = torch.sigmoid(outputs)
        all_preds.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        pbar.set_postfix({"loss": loss.item()})

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)

    return epoch_loss, metrics


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            # Use autocast for validation as well if on GPU
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)

    return epoch_loss, metrics


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def train_pipeline(model, train_loader, val_loader, config, device, save_dir):
    criterion = nn.BCEWithLogitsLoss()

    # Print trainable info
    print_trainable_parameters(model)

    # Filter params to only those requiring grad
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=config["training"]["learning_rate"])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2
    )

    # Initialize AMP Scaler if on CUDA
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    if scaler:
        print("Enabling Mixed Precision (AMP) training.")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    early_stopping_counter = 0
    patience = config["training"]["patience"]

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"Starting training on {device}...")

    for epoch in range(config["training"]["num_epochs"]):
        print(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler
        )
        val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_loss:.4f} Acc: {val_metrics['accuracy']:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        # Learning Rate Scheduling
        scheduler.step(val_loss)

        # Early Stopping & Checkpointing
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stopping_counter = 0

            # Save best model
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print("Model Checkpoint Saved!")
        else:
            early_stopping_counter += 1
            print(f"Early Stopping Counter: {early_stopping_counter}/{patience}")

        if early_stopping_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load best weights
    model.load_state_dict(best_model_wts)

    # Save History
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f)

    return model, history
