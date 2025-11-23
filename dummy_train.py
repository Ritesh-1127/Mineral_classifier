import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import os, csv, random, shutil
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# matplotlib (headless-safe)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ⚙️ CUDA setup
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

# -----------------------------
# 🪨 Paths (LOCAL VERSION)
# -----------------------------
orig_train_dir = "explo_data/data/data"
orig_val_dir = "explo_data/mineral-trier"

balanced_train_dir = "balanced_train"
balanced_val_dir = "balanced_val"

# NEW checkpoint/log folder (keeps original checkpoints untouched)
ckpt_path = "checkpoints_new_b4/mineralnet_b4_last.pth"
log_path = "checkpoints_new_b4/training_log_b4.csv"

# -----------------------------
# ⚙️ Hyperparams
# -----------------------------
# NOTE: 380x380 increases memory. If you get OOM, reduce batch_size.
IMG_SIZE = 380
batch_size = 16
num_epochs = 9
learning_rate = 1e-4
weight_decay = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TRAIN = 15000
MAX_VAL = 10000

# -----------------------------
# ⚖️ Trim + Rebuild Datasets
# -----------------------------
import hashlib

SMART_SAMPLING = True

def trim_and_copy(src_dir, dest_dir, max_samples, name):
    os.makedirs(dest_dir, exist_ok=True)
    for cls in tqdm(sorted(os.listdir(src_dir))):
        cls_path = os.path.join(src_dir, cls)

        # Skip fluorite completely
        if cls.lower() == "fluorite":
            continue

        if not os.path.isdir(cls_path):
            continue

        imgs = [os.path.join(cls_path, f) for f in os.listdir(cls_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        dest_cls_path = os.path.join(dest_dir, cls)
        os.makedirs(dest_cls_path, exist_ok=True)

        hash_now = hashlib.md5("".join(sorted(imgs)).encode()).hexdigest()
        hash_file = os.path.join(dest_cls_path, "_hash.txt")

        prev_hash = ""
        if os.path.exists(hash_file):
            with open(hash_file, "r") as f:
                prev_hash = f.read().strip()

        if SMART_SAMPLING and hash_now == prev_hash:
            existing = len(os.listdir(dest_cls_path)) - 1 if os.path.exists(dest_cls_path) else 0
            if existing >= min(len(imgs), max_samples):
                continue

        if len(imgs) > max_samples:
            imgs = random.sample(imgs, max_samples)

        # Clear old class folder
        for old in os.listdir(dest_cls_path):
            os.remove(os.path.join(dest_cls_path, old))

        # Copy fresh images
        for img in imgs:
            shutil.copy(img, os.path.join(dest_cls_path, os.path.basename(img)))

        # Save hash
        with open(hash_file, "w") as f:
            f.write(hash_now)

# Always rebuild with smart logic
trim_and_copy(orig_train_dir, balanced_train_dir, MAX_TRAIN, "training")
trim_and_copy(orig_val_dir, balanced_val_dir, MAX_VAL, "validation")

# -----------------------------
# 🧠 Data Transforms
# -----------------------------
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# 💾 Checkpoint Utils
# -----------------------------
def save_checkpoint(model, optimizer, epoch, acc, path=ckpt_path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "accuracy": acc,
    }, path)
    print(f"💾 Saved checkpoint: {path}")

def load_checkpoint(model, optimizer, path=ckpt_path):
    if os.path.exists(path):
        try:
            ckpt = torch.load(path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_acc = ckpt.get("accuracy", 0.0)
            print(f"🔁 Loaded checkpoint from epoch {start_epoch-1} | Acc: {best_acc:.2f}%")
            return start_epoch, best_acc
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint:\n{e}")
    print("🚫 No checkpoint found, starting fresh.")
    return 0, 0.0

# -----------------------------
# 🧩 Model (EfficientNet-B4)
# -----------------------------
def build_model(num_classes):
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    # replace final linear
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

# -----------------------------
# 📊 Plot helpers (write to checkpoints_new_b4)
# -----------------------------
def plot_training_lists(train_losses, val_accuracies, lrs=None, out_png='checkpoints_new_b4/training_plot_live.png'):
    if not train_losses:
        return
    epochs = list(range(1, len(train_losses) + 1))
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(epochs, train_losses, marker='o', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_accuracies, marker='x', label='Val Acc')
    ax2.set_ylabel('Val Accuracy (%)')

    if lrs:
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        ax3.plot(epochs, lrs, linestyle='--', label='LR')
        ax3.set_ylabel('LR')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    extra_lines, extra_labels = (ax3.get_legend_handles_labels() if lrs else ([], []))
    ax1.legend(lines + lines2 + extra_lines, labels + labels2 + extra_labels, loc='best')
    plt.title('Train Loss & Val Accuracy')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png)
    plt.close(fig)

def plot_training_log(log_path='checkpoints_new_b4/training_log_b4.csv', out_png='checkpoints_new_b4/training_plot_b4.png'):
    if not os.path.exists(log_path):
        print(f"🚫 Log file not found: {log_path}")
        return

    epochs, train_loss, val_acc, lr = [], [], [], []
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(row.get('Epoch', len(epochs)+1)))
                train_loss.append(float(row.get('Train_Loss', row.get('Loss', 0))))
                val_acc.append(float(row.get('Val_Acc', row.get('ValAccuracy', 0))))
                lr.append(float(row.get('LR', row.get('LearningRate', 0))))
            except Exception:
                continue

    if not epochs:
        print("⚠️ No data parsed from log.")
        return

    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(epochs, train_loss, marker='o', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_acc, marker='x', label='Val Accuracy')
    ax2.set_ylabel('Val Accuracy (%)')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')

    plt.title('Training Loss & Val Accuracy (from CSV)')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png)
    plt.close(fig)
    print(f"✅ Saved training plot to: {out_png}")

# -----------------------------
# 🧪 Training Loop
# -----------------------------
def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs, start_epoch=0):
    best_acc = 0.0
    patience, trigger = 6, 0
    scaler = torch.amp.GradScaler('cuda')

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["Epoch", "Train_Loss", "Val_Acc", "LR"])

    # lists for live plotting
    train_losses = []
    val_accuracies = []
    lrs = []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for imgs, targets in pbar:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                if isinstance(outputs, tuple):
                    main_output, aux_output = outputs
                    loss = criterion(main_output, targets) + 0.4 * criterion(aux_output, targets)
                else:
                    loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = running_loss / max(1, len(train_loader))
        print(f"✅ Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.inference_mode():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        acc = 100 * correct / max(1, total)
        print(f"📈 Val Accuracy: {acc:.2f}%")

        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else None

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, avg_loss, acc, current_lr])

        # append for live plot
        train_losses.append(avg_loss)
        val_accuracies.append(acc)
        lrs.append(current_lr if current_lr is not None else 0.0)

        # save a live-ish plot each epoch
        plot_training_lists(train_losses, val_accuracies, lrs, out_png='checkpoints_new_b4/training_plot_live.png')

        save_checkpoint(model, optimizer, epoch, acc)
        if acc > best_acc:
            best_acc = acc
            trigger = 0
            torch.save(model.state_dict(), os.path.join(os.path.dirname(ckpt_path), "mineralnet_b4_best.pth"))
            print(f"🏅 New Best: {best_acc:.2f}%")
        else:
            trigger += 1
            if trigger >= patience:
                print("🛑 Early stopping triggered.")
                break

# -----------------------------
# 🚀 Entry Point
# -----------------------------
if __name__ == "__main__":
    from torch import multiprocessing
    multiprocessing.freeze_support()

    train_ds = datasets.ImageFolder(balanced_train_dir, transform=train_transforms)
    val_ds = datasets.ImageFolder(balanced_val_dir, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(num_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4, epochs=num_epochs, steps_per_epoch=max(1, len(train_loader))
    )

    start_epoch, _ = load_checkpoint(model, optimizer)
    train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs, start_epoch)

    # once done (or if you already trained earlier), create final plot from CSV:
    plot_training_log(log_path=log_path)

