"""Convert a YOLO-format detection dataset (images + .txt label files — what Roboflow exports
as "YOLOv8") into the ImageFolder crop layout train_emergency.py expects:
    dataset/train/ambulance|fire_engine|normal/*.jpg
    dataset/val/ambulance|fire_engine|normal/*.jpg

Edit CLASS_MAP below to match the class names in YOUR downloaded dataset's data.yaml (print
it with `python -c "import yaml; print(yaml.safe_load(open('data.yaml'))['names'])"` if unsure)
and which of our 3 buckets each one belongs in. Anything not listed is skipped — that's how you
drop classes you don't want (e.g. a "police_car" class you'd rather not train on at all).

Expected input layout (standard Roboflow YOLOv8 export):
    src/
      train/images/*.jpg   train/labels/*.txt
      valid/images/*.jpg   valid/labels/*.txt   (Roboflow calls it "valid"; this script maps it to "val")
      data.yaml

Run:
    pip install pyyaml opencv-python
    python crop_dataset.py --src path/to/downloaded/dataset --out dataset
    python train_emergency.py --data dataset --epochs 12 --out models/emergency_cnn.pt
"""
import argparse
import glob
import os

import cv2
import yaml

# ---- EDIT THIS to match your downloaded dataset's actual class names (case-sensitive) ----
CLASS_MAP = {
    "ambulance": "ambulance", "Ambulance": "ambulance",
    "fire_truck": "fire_engine", "Fire_truck": "fire_engine", "fire truck": "fire_engine",
    "firetruck": "fire_engine", "Firetruck": "fire_engine", "fire_engine": "fire_engine",
    "car": "normal", "Car": "normal", "bus": "normal", "Bus": "normal", "truck": "normal", "Truck": "normal",
    "normal vehicle": "normal", "Normal Vehicle": "normal", "normal_vehicle": "normal",
    "police_car": "normal", "Police_car": "normal", "news_van": "normal", "News_van": "normal",
}


def crop_split(src, src_split, out_split, out_root, names):
    img_dir = os.path.join(src, src_split, "images")
    lbl_dir = os.path.join(src, src_split, "labels")
    if not os.path.isdir(img_dir):
        return 0
    n = 0
    for img_path in glob.glob(os.path.join(img_dir, "*")):
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, base + ".txt")
        if not os.path.exists(lbl_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        for i, line in enumerate(open(lbl_path)):
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            cname = names[cid] if cid < len(names) else None
            bucket = CLASS_MAP.get(cname)
            if not bucket:
                continue
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            out_dir = os.path.join(out_root, out_split, bucket)
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, f"{base}_{i}.jpg"), crop)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="downloaded Roboflow YOLOv8-format dataset root (has data.yaml)")
    ap.add_argument("--out", default="dataset", help="output ImageFolder root for train_emergency.py")
    args = ap.parse_args()

    with open(os.path.join(args.src, "data.yaml")) as f:
        names = yaml.safe_load(f)["names"]
    print("Source classes:", names)

    total = 0
    for src_split, out_split in (("train", "train"), ("valid", "val"), ("val", "val"), ("test", "val")):
        n = crop_split(args.src, src_split, out_split, args.out, names)
        if n:
            print(f"{src_split} -> {out_split}: {n} crops")
        total += n

    if total == 0:
        raise SystemExit("No crops written — check --src points at the dataset root (the folder with data.yaml).")
    print(f"Done: {total} crops under {args.out}/. Now run train_emergency.py --data {args.out}")


if __name__ == "__main__":
    main()
