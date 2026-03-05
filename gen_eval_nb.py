#!/usr/bin/env python3
"""Generates eval.ipynb — Food-11 test-set evaluation notebook."""
import json
from pathlib import Path

OUT = Path('/scratch-share/QIAO0042/models/ineedfood/eval.ipynb')

def md(source, cid):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": source}

def code(source, cid):
    return {"cell_type": "code", "execution_count": None, "id": cid,
            "metadata": {}, "outputs": [], "source": source}

cells = []

cells.append(md(
    "# Food-11 — Test Set Evaluation\n\n"
    "Loads a saved checkpoint and evaluates on the hold-out test set.\n"
    "Reports overall accuracy, per-class accuracy, and confusion matrix.",
    "eval-title"
))

# ── Setup ──────────────────────────────────────────────────────────────────────
cells.append(code(
    "import os, json\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "from pathlib import Path\n\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "from torch.utils.data import DataLoader, Dataset\n"
    "import torchvision.transforms as T\n"
    "from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights\n"
    "from PIL import Image\n\n"
    "ROOT = os.path.expanduser('~/QIAO0042/models/ineedfood/')\n"
    "os.chdir(ROOT)\n\n"
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
    "print(f'Device: {DEVICE}')",
    "eval-setup"
))

cells.append(code(
    "with open('dataset/stats.json') as f:\n"
    "    _stats = json.load(f)\n"
    "MEAN        = _stats['mean']\n"
    "STD         = _stats['std']\n"
    "NUM_CLASSES = _stats['num_classes']\n\n"
    "CLASS_NAMES = [\n"
    "    'Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food',\n"
    "    'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit'\n"
    "]\n"
    "assert len(CLASS_NAMES) == NUM_CLASSES\n"
    "print(f'Classes: {NUM_CLASSES}  |  mean={[f\"{v:.3f}\" for v in MEAN]}')",
    "eval-stats"
))

cells.append(code(
    "LOCAL_CACHE = '/tmp/ineedfood_100x100'\n"
    "TEST_DIR    = str(Path(LOCAL_CACHE) / 'evaluation')\n\n"
    "if not Path(TEST_DIR).exists():\n"
    "    import shutil\n"
    "    NFS_CACHE = _stats['nfs_cache']\n"
    "    src = Path(NFS_CACHE) / 'evaluation'\n"
    "    dst = Path(TEST_DIR)\n"
    "    print(f'Copying evaluation split to /tmp ...')\n"
    "    shutil.copytree(str(src), str(dst))\n"
    "    print('Done.')\n"
    "else:\n"
    "    print(f'Test dir ready: {TEST_DIR}')",
    "eval-cache"
))

cells.append(code(
    "class Food11Dataset(Dataset):\n"
    "    def __init__(self, root_dir, transform=None):\n"
    "        self.root = Path(root_dir)\n"
    "        self.transform = transform\n"
    "        self.samples = []\n"
    "        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}\n"
    "        for p in self.root.iterdir():\n"
    "            if p.is_file() and p.suffix.lower() in exts:\n"
    "                stem = p.stem\n"
    "                if '_' not in stem: continue\n"
    "                cls = int(stem.split('_')[0])\n"
    "                self.samples.append((p, cls))\n"
    "        if not self.samples:\n"
    "            raise RuntimeError(f'No images in {root_dir}')\n"
    "        self.classes = [str(i) for i in range(11)]\n\n"
    "    def __len__(self): return len(self.samples)\n\n"
    "    def __getitem__(self, idx):\n"
    "        path, label = self.samples[idx]\n"
    "        img = Image.open(path).convert('RGB')\n"
    "        if self.transform: img = self.transform(img)\n"
    "        return img, label\n\n\n"
    "EVAL_TRANSFORM = T.Compose([\n"
    "    T.ToTensor(),\n"
    "    T.Normalize(mean=MEAN, std=STD),\n"
    "])\n\n"
    "test_dataset = Food11Dataset(TEST_DIR, transform=EVAL_TRANSFORM)\n"
    "test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False,\n"
    "                          num_workers=8, pin_memory=True)\n"
    "print(f'Test set: {len(test_dataset)} images')",
    "eval-dataset"
))

cells.append(code(
    "def build_model(num_classes=NUM_CLASSES):\n"
    "    model = efficientnet_b0(weights=None)\n"
    "    in_f = model.classifier[1].in_features\n"
    "    model.classifier[1] = nn.Linear(in_f, num_classes)\n"
    "    return model.to(DEVICE)\n\n\n"
    "_AMP_CTX = dict(device_type=DEVICE.type, dtype=torch.bfloat16,\n"
    "                enabled=False)\n\n"
    "print('Model factory ready.')",
    "eval-model-fn"
))

# ── Load checkpoint ────────────────────────────────────────────────────────────
cells.append(md(
    "## Load Checkpoint\n\n"
    "Set `CKPT_PATH` to the checkpoint you want to evaluate.\n"
    "Checkpoints are saved in `checkpoints/` by `train.ipynb`.",
    "eval-ckpt-hdr"
))

cells.append(code(
    "CKPT_PATH = 'checkpoints/sec6_mixup_best.pth'  # ← change as needed\n\n"
    "ckpt = torch.load(CKPT_PATH, map_location=DEVICE)\n"
    "model = build_model()\n"
    "model.load_state_dict(ckpt['model_state_dict'])\n"
    "model.eval()\n\n"
    "print(f'Loaded: {CKPT_PATH}')\n"
    "print(f'  Saved at epoch : {ckpt[\"epoch\"]}')\n"
    "print(f'  Val acc        : {ckpt[\"val_acc\"]:.4f}')\n"
    "print(f'  Val loss       : {ckpt[\"val_loss\"]:.4f}')",
    "eval-load-ckpt"
))

# ── Overall accuracy ───────────────────────────────────────────────────────────
cells.append(md("## Overall Test Accuracy", "eval-overall-hdr"))

cells.append(code(
    "@torch.no_grad()\n"
    "def evaluate_full(model, loader):\n"
    "    \"\"\"Returns loss, accuracy, all predictions, and all labels.\"\"\"\n"
    "    model.eval()\n"
    "    criterion = nn.CrossEntropyLoss()\n"
    "    total_loss, correct, total = 0.0, 0, 0\n"
    "    all_preds, all_labels = [], []\n"
    "    for imgs, labels in loader:\n"
    "        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)\n"
    "        with torch.autocast(**_AMP_CTX):\n"
    "            logits = model(imgs)\n"
    "            loss   = criterion(logits, labels)\n"
    "        preds = logits.argmax(1)\n"
    "        total_loss += loss.item() * imgs.size(0)\n"
    "        correct    += (preds == labels).sum().item()\n"
    "        total      += imgs.size(0)\n"
    "        all_preds.append(preds.cpu())\n"
    "        all_labels.append(labels.cpu())\n"
    "    all_preds  = torch.cat(all_preds)\n"
    "    all_labels = torch.cat(all_labels)\n"
    "    return total_loss / total, correct / total, all_preds, all_labels\n\n\n"
    "test_loss, test_acc, preds, labels = evaluate_full(model, test_loader)\n"
    "print(f'Test loss : {test_loss:.4f}')\n"
    "print(f'Test acc  : {test_acc:.4f}  ({test_acc*100:.2f}%)')",
    "eval-overall"
))

# ── Per-class accuracy ─────────────────────────────────────────────────────────
cells.append(md("## Per-Class Accuracy", "eval-perclass-hdr"))

cells.append(code(
    "per_class_correct = torch.zeros(NUM_CLASSES)\n"
    "per_class_total   = torch.zeros(NUM_CLASSES)\n"
    "for p, l in zip(preds, labels):\n"
    "    per_class_total[l]   += 1\n"
    "    per_class_correct[l] += (p == l).float()\n\n"
    "per_class_acc = (per_class_correct / per_class_total.clamp(min=1)).numpy()\n\n"
    "print(f'{\"Class\":<20} {\"N\":>5}  {\"Acc\":>7}')\n"
    "print('-' * 36)\n"
    "for i, (name, acc, n) in enumerate(zip(CLASS_NAMES, per_class_acc, per_class_total)):\n"
    "    bar = '#' * int(acc * 20)\n"
    "    print(f'{name:<20} {int(n):>5}  {acc:>6.2%}  {bar}')\n"
    "print('-' * 36)\n"
    "print(f'{\"Overall\":<20} {len(labels):>5}  {test_acc:>6.2%}')",
    "eval-perclass"
))

cells.append(code(
    "fig, ax = plt.subplots(figsize=(10, 4))\n"
    "colors = ['#d62728' if a < 0.65 else '#2ca02c' for a in per_class_acc]\n"
    "bars = ax.bar(CLASS_NAMES, per_class_acc * 100, color=colors, edgecolor='white')\n"
    "ax.axhline(test_acc * 100, color='steelblue', linestyle='--', linewidth=1.5,\n"
    "           label=f'Overall {test_acc*100:.1f}%')\n"
    "ax.set_ylabel('Accuracy (%)')\n"
    "ax.set_title('Per-Class Test Accuracy — Food-11')\n"
    "ax.set_ylim(0, 105)\n"
    "ax.legend()\n"
    "plt.xticks(rotation=30, ha='right')\n"
    "for bar, acc in zip(bars, per_class_acc):\n"
    "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,\n"
    "            f'{acc:.0%}', ha='center', va='bottom', fontsize=8)\n"
    "plt.tight_layout()\n"
    "plt.savefig('per_class_acc.png', dpi=150)\n"
    "plt.show()",
    "eval-perclass-plot"
))

# ── Confusion matrix ───────────────────────────────────────────────────────────
cells.append(md("## Confusion Matrix", "eval-cm-hdr"))

cells.append(code(
    "cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)\n"
    "for p, l in zip(preds, labels):\n"
    "    cm[l, p] += 1\n\n"
    "# Normalise by true class (row-wise)\n"
    "cm_norm = cm.float() / cm.sum(dim=1, keepdim=True).clamp(min=1)\n\n"
    "fig, ax = plt.subplots(figsize=(10, 8))\n"
    "im = ax.imshow(cm_norm.numpy(), cmap='Blues', vmin=0, vmax=1)\n"
    "plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)\n\n"
    "ax.set_xticks(range(NUM_CLASSES))\n"
    "ax.set_yticks(range(NUM_CLASSES))\n"
    "ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=8)\n"
    "ax.set_yticklabels(CLASS_NAMES, fontsize=8)\n"
    "ax.set_xlabel('Predicted')\n"
    "ax.set_ylabel('True')\n"
    "ax.set_title('Confusion Matrix (row-normalised) — Food-11 Test Set')\n\n"
    "for i in range(NUM_CLASSES):\n"
    "    for j in range(NUM_CLASSES):\n"
    "        val = cm_norm[i, j].item()\n"
    "        color = 'white' if val > 0.5 else 'black'\n"
    "        ax.text(j, i, f'{val:.2f}', ha='center', va='center',\n"
    "                fontsize=6, color=color)\n\n"
    "plt.tight_layout()\n"
    "plt.savefig('confusion_matrix.png', dpi=150)\n"
    "plt.show()",
    "eval-cm"
))

# ── Top confusions ─────────────────────────────────────────────────────────────
cells.append(md("## Top Confusions", "eval-topconf-hdr"))

cells.append(code(
    "# Show most common misclassification pairs\n"
    "print(f'{\"True Class\":<20} {\"Predicted As\":<20} {\"Count\":>5}  {\"Rate\":>7}')\n"
    "print('-' * 58)\n"
    "off_diag = []\n"
    "for i in range(NUM_CLASSES):\n"
    "    for j in range(NUM_CLASSES):\n"
    "        if i != j and cm[i, j] > 0:\n"
    "            rate = cm_norm[i, j].item()\n"
    "            off_diag.append((int(cm[i, j]), rate, i, j))\n"
    "off_diag.sort(reverse=True)\n"
    "for count, rate, i, j in off_diag[:10]:\n"
    "    print(f'{CLASS_NAMES[i]:<20} {CLASS_NAMES[j]:<20} {count:>5}  {rate:>6.1%}')",
    "eval-topconf"
))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells
}

with open(OUT, 'w') as f:
    json.dump(nb, f, indent=1)

print(f'Written: {OUT}  ({len(cells)} cells)')
