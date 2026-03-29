import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from tqdm import tqdm
import time
import copy
from utils.metrics import compute_metrics, save_metrics


from utils.gpu_preprocessing import GPUFracturePreprocessor

from utils.monitoring import ResourceMonitor


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    scaler=None,
    gpu_pre=None,
    accumulation_steps=1,
    monitor=None,
):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Training")
    optimizer.zero_grad(set_to_none=True)

    for i, (inputs, labels) in enumerate(pbar):
        # non_blocking=True is faster when used with pin_memory=True
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

        # ONLY GPU: Final heavy preprocessing happens here on the graphics card
        if gpu_pre is not None:
            with torch.amp.autocast("cuda"):  # Ensure preprocessor also uses AMP
                inputs = gpu_pre(inputs)

        # Automatic Mixed Precision (AMP)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * inputs.size(0) * accumulation_steps

        # Sigmoid for binary predictions
        probs = torch.sigmoid(outputs)
        all_preds.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        pbar.set_postfix({"loss": loss.item() * accumulation_steps})

        if monitor:
            monitor.step()

        # Explicitly delete large tensors to help 4GB VRAM
        del inputs, labels, outputs, loss

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)

    return epoch_loss, metrics


def validate_one_epoch(model, dataloader, criterion, device, gpu_pre=None):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

            if gpu_pre is not None:
                inputs = gpu_pre(inputs)

            # Use autocast for validation as well if on GPU
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            del inputs, labels, outputs, loss

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


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        f_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return f_loss.mean()


class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.1, pos_weight=None):
        super(LabelSmoothingBCE, self).__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, inputs, targets):
        with torch.no_grad():
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(inputs, targets)


def train_pipeline(model, train_loader, val_loader, config, device, save_dir):
    # 1. OPTIMIZATION: Enable cuDNN auto-tuner
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("CUDA Optimization: cuDNN Benchmark enabled.")

    # 2. RESOURCE MONITORING & ADAPTIVE OPTIMIZATION
    monitor = ResourceMonitor(interval_batches=20)
    accumulation_steps = config.get("training", {}).get("accumulation_steps", 1)

    if device.type == "cuda":
        total_vram = torch.cuda.get_device_properties(device).total_memory / (
            1024**3
        )  # GB
        print(f"Detected GPU with {total_vram:.2f} GB VRAM.")

        # Auto-adjust if VRAM is low and not explicitly set
        if total_vram < 5.0 and accumulation_steps == 1:
            accumulation_steps = 4
            print(
                f"LOW VRAM DETECTED (<5GB). Enabling Gradient Accumulation (steps={accumulation_steps}) to prevent OOM."
            )
        elif total_vram < 8.0 and accumulation_steps == 1:
            accumulation_steps = 2
            print(
                f"MODERATE VRAM DETECTED (<8GB). Enabling Gradient Accumulation (steps={accumulation_steps})."
            )

    # LOSS CHOICE
    loss_type = config.get("training", {}).get("loss", "bce")
    
    # Calculate pos_weight
    pos_weight = None
    try:
        from collections import Counter
        labels = [item[1] for item in train_loader.dataset.samples]
        counts = Counter(labels)
        if counts[1] > 0:
            pos_weight = torch.tensor([counts[0] / counts[1]]).to(device)
    except: pass

    if loss_type == "focal":
        criterion = FocalLoss()
        print("Loss: Using Focal Loss for hard-sample focus.")
    else:
        # Optimization: Use Label Smoothing for better generalization
        criterion = LabelSmoothingBCE(smoothing=0.1, pos_weight=pos_weight)
        print(f"Loss: Using LabelSmoothingBCE (smoothing=0.1, pos_weight={pos_weight.item() if pos_weight else None}).")

    # Optimize Model Memory Layout
    if device.type == "cuda":
        model = model.to(device, memory_format=torch.channels_last)
    else:
        model = model.to(device)

    gpu_pre = None
    if device.type == "cuda":
        simple_pre = config.get("dataset", {}).get("simple_pre", False)
        gpu_pre = GPUFracturePreprocessor(device=device, simple_pre=simple_pre).to(
            device, memory_format=torch.channels_last
        )

    # OPTIMIZER CHOICE
    lr = config["training"]["learning_rate"]
    opt_type = config.get("training", {}).get("optimizer", "adam")

    def get_optimizer(model, lr, stage=0):
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        if stage == 0:
            return optim.Adam(trainable_params, lr=lr)
        else:
            # Lower LR for backbones, higher for new heads
            backbone_params = []
            head_params = []
            for name, param in model.named_parameters():
                if not param.requires_grad: continue
                if "cnn" in name or "vit" in name:
                    backbone_params.append(param)
                else: head_params.append(param)
            
            return optim.Adam([
                {"params": backbone_params, "lr": lr * 0.1},
                {"params": head_params, "lr": lr}
            ])

    optimizer = get_optimizer(model, lr, stage=0)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2
    )

    # Initialize AMP Scaler if on CUDA
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    if scaler:
        print("Enabling Mixed Precision (AMP) training for GPU optimization.")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    early_stopping_counter = 0
    patience = config["training"]["patience"]

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(
        f"Starting training on {device} with Accumulation Steps: {accumulation_steps}"
    )

    for epoch in range(config["training"]["num_epochs"]):
        # Gradual Unfreezing
        if hasattr(model, "unfreeze_stage"):
            stage_updated = False
            if epoch == 3:
                model.unfreeze_stage(1)
                stage_updated = True
                new_lr = config["training"]["learning_rate"] * 0.1
            elif epoch == 6:
                model.unfreeze_stage(2)
                stage_updated = True
                new_lr = config["training"]["learning_rate"] * 0.05

            if stage_updated:
                optimizer = get_optimizer(model, new_lr, stage=1 if epoch == 3 else 2)
                print(f"Unfroze backbone stage! Rebuilt optimizer with lower backbone LR.")

        # Optional: Clear VRAM cache at start of epoch
        if device.type == "cuda":
            torch.cuda.empty_cache()
            vram_mb = torch.cuda.memory_reserved() / (1024**2)
            print(f"VRAM Reserved: {vram_mb:.2f} MB")

        print(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")

        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            gpu_pre=gpu_pre,
            accumulation_steps=accumulation_steps,
            monitor=monitor,
        )
        val_loss, val_metrics = validate_one_epoch(
            model, val_loader, criterion, device, gpu_pre=gpu_pre
        )

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
    monitor.final_report()

    # Save History
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f)

    return model, history
