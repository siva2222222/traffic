"""Fine-tune a small MobileNetV2 classifier to tell ambulance / fire_engine / normal-vehicle
crops apart. This is a CLASSIFIER (one label per cropped image), not a YOLO object detector,
so it's small and fast to train — the backbone is frozen, only the final layer trains, which
works fine on CPU for a modest dataset (though a free Colab GPU is much faster for a bigger
one; see TRAIN_EMERGENCY_MODEL.md for a ready-to-paste Colab version of this).

Expected folder layout (ImageFolder — class-folder NAMES AND ORDER must match
emergency_classifier.py's CLASSES, which is alphabetical: ambulance, fire_engine, normal):

    dataset/
      train/
        ambulance/*.jpg
        fire_engine/*.jpg
        normal/*.jpg
      val/
        ambulance/*.jpg
        fire_engine/*.jpg
        normal/*.jpg

Where to get images: public datasets with ambulance + fire-truck classes exist on Roboflow
Universe (search "emergency vehicle detection" / "ambulance firetruck") — see
TRAIN_EMERGENCY_MODEL.md for specific dataset links. You want roughly similar counts of
normal-vehicle crops too (your own project's `static/evidence/` folder, or any car/bus/truck
images) so the classifier isn't just trained to say "yes" to everything.

Run:
    pip install torch torchvision
    python train_emergency.py --data dataset --epochs 12 --out models/emergency_cnn.pt
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

EXPECTED_CLASSES = ["ambulance", "fire_engine", "normal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root with train/ and val/ subfolders (ImageFolder layout)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models/emergency_cnn.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    tfm_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tfm_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(args.data, "train"), tfm_train)
    val_ds = datasets.ImageFolder(os.path.join(args.data, "val"), tfm_val)
    print("Detected classes:", train_ds.classes)
    if train_ds.classes != EXPECTED_CLASSES:
        raise SystemExit(
            f"Folder names/order must be exactly {EXPECTED_CLASSES} (ImageFolder sorts "
            f"alphabetically) — got {train_ds.classes}. Rename your dataset/train and "
            f"dataset/val subfolders to match."
        )

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in model.features.parameters():          # freeze backbone - fast, works fine on CPU
        p.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, len(EXPECTED_CLASSES))
    model.to(device)

    opt = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        train_loss = running / max(1, len(train_ds))

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
        val_acc = correct / max(1, len(val_ds))
        print(f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.3f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.out)
            print(f"  -> saved {args.out} (best val_acc so far: {best_acc:.3f})")

    print(f"Done. Best val_acc={best_acc:.3f}. Weights at {args.out}")
    print("Drop that file in as models/emergency_cnn.pt in the main project — engine.py picks it up automatically.")


if __name__ == "__main__":
    main()
