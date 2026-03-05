#!/usr/bin/env python3
"""Generates mixup_visualization.ipynb"""
import json
from pathlib import Path

OUT = Path('/scratch-share/QIAO0042/models/ineedfood/mixup_visualization.ipynb')

def md(source, cid):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": source}

def code(source, cid):
    return {"cell_type": "code", "execution_count": None, "id": cid,
            "metadata": {}, "outputs": [], "source": source}

cells = []

cells.append(md(
    "# Mixup Augmentation — Visualisation\n\n"
    "Three views:\n"
    "1. **Beta distribution** — how λ is sampled for α=0.2\n"
    "2. **Image grid** — pairs of Food-11 images mixed at varying λ\n"
    "3. **Alpha sweep** — effect of different α values on the same image pair",
    "mv-title"
))

# ── Setup ──────────────────────────────────────────────────────────────────────
cells.append(code(
    "import os, json, random\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.gridspec as gridspec\n"
    "from pathlib import Path\n\n"
    "import torch\n"
    "import torchvision.transforms as T\n"
    "from PIL import Image\n"
    "from scipy.stats import beta as beta_dist\n\n"
    "ROOT = os.path.expanduser('~/QIAO0042/models/ineedfood/')\n"
    "os.chdir(ROOT)\n\n"
    "with open('dataset/stats.json') as f:\n"
    "    _stats = json.load(f)\n"
    "MEAN = _stats['mean']\n"
    "STD  = _stats['std']\n\n"
    "CLASS_NAMES = [\n"
    "    'Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food',\n"
    "    'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit'\n"
    "]\n\n"
    "TRAIN_DIR = '/tmp/ineedfood_100x100/training'\n"
    "random.seed(42)\n"
    "print('Setup done.')",
    "mv-setup"
))

# ── Helper: load images ────────────────────────────────────────────────────────
cells.append(code(
    "def load_images(train_dir, n=16):\n"
    "    \"\"\"Load n random images with labels from the flat training dir.\"\"\"\n"
    "    paths = list(Path(train_dir).glob('*.jpg'))\n"
    "    chosen = random.sample(paths, n)\n"
    "    imgs, labels = [], []\n"
    "    for p in chosen:\n"
    "        cls = int(p.stem.split('_')[0])\n"
    "        img = Image.open(p).convert('RGB')\n"
    "        imgs.append(img)\n"
    "        labels.append(cls)\n"
    "    return imgs, labels\n\n\n"
    "def to_tensor(img):\n"
    "    return T.ToTensor()(img)   # (3, H, W), float32 in [0,1]\n\n\n"
    "def to_display(tensor):\n"
    "    \"\"\"Convert tensor (3,H,W) in [0,1] to numpy (H,W,3) for imshow.\"\"\"\n"
    "    return tensor.permute(1, 2, 0).clamp(0, 1).numpy()\n\n\n"
    "def mixup(img_a, img_b, lam):\n"
    "    ta, tb = to_tensor(img_a), to_tensor(img_b)\n"
    "    return lam * ta + (1 - lam) * tb\n\n\n"
    "imgs, labels = load_images(TRAIN_DIR, n=20)\n"
    "print(f'Loaded {len(imgs)} images')",
    "mv-helpers"
))

# ── Plot 1: Beta distribution ──────────────────────────────────────────────────
cells.append(md(
    "## 1 — Beta Distribution\n\n"
    "Shows how λ is sampled for different α values. "
    "α < 1 → U-shaped (most mixes are close to one original). "
    "α = 1 → uniform. α > 1 → bell-shaped (mixes cluster near 0.5).",
    "mv-beta-hdr"
))

cells.append(code(
    "alphas  = [0.1, 0.2, 0.5, 1.0, 2.0]\n"
    "colors  = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']\n"
    "x       = np.linspace(1e-3, 1 - 1e-3, 500)\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n\n"
    "# Left: PDF curves\n"
    "for a, c in zip(alphas, colors):\n"
    "    pdf = beta_dist.pdf(x, a=a, b=a)\n"
    "    axes[0].plot(x, pdf, color=c, linewidth=2, label=f'α={a}')\n"
    "axes[0].set_xlim(0, 1)\n"
    "axes[0].set_ylim(0, 6)\n"
    "axes[0].set_xlabel('λ (mixing coefficient)', fontsize=11)\n"
    "axes[0].set_ylabel('Density', fontsize=11)\n"
    "axes[0].set_title('Beta(α, α) PDF — sampling distribution of λ', fontsize=11)\n"
    "axes[0].legend(fontsize=10)\n"
    "axes[0].axvline(0.5, color='gray', linestyle=':', linewidth=1)\n\n"
    "# Right: histogram of 5000 samples for α=0.2\n"
    "samples = np.random.beta(0.2, 0.2, 5000)\n"
    "axes[1].hist(samples, bins=50, color='#377eb8', alpha=0.7, edgecolor='white')\n"
    "axes[1].set_xlabel('λ', fontsize=11)\n"
    "axes[1].set_ylabel('Count', fontsize=11)\n"
    "axes[1].set_title('Empirical distribution of λ  (α=0.2, n=5000)', fontsize=11)\n"
    "pct_extreme = ((samples < 0.1) | (samples > 0.9)).mean()\n"
    "axes[1].set_title(\n"
    "    f'Empirical λ distribution  (α=0.2) — {pct_extreme:.0%} of draws outside [0.1, 0.9]',\n"
    "    fontsize=10)\n\n"
    "plt.tight_layout()\n"
    "plt.savefig('mixup_beta_dist.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "print(f'Saved mixup_beta_dist.png')",
    "mv-beta-plot"
))

# ── Plot 2: Image grid ─────────────────────────────────────────────────────────
cells.append(md(
    "## 2 — Mixed Image Grid\n\n"
    "Each row shows one image pair mixed at four λ values: 0.75, 0.5, 0.25, and a "
    "random draw from Beta(0.2, 0.2). "
    "The label above each mixed image shows `λ·A + (1-λ)·B`.",
    "mv-grid-hdr"
))

cells.append(code(
    "N_PAIRS = 4\n"
    "LAMBDAS = [0.75, 0.5, 0.25]\n\n"
    "# Pick N_PAIRS non-same-class pairs\n"
    "pairs = []\n"
    "idx   = list(range(len(imgs)))\n"
    "random.shuffle(idx)\n"
    "for i in range(0, len(idx) - 1, 2):\n"
    "    a, b = idx[i], idx[i+1]\n"
    "    if labels[a] != labels[b]:\n"
    "        pairs.append((a, b))\n"
    "    if len(pairs) == N_PAIRS:\n"
    "        break\n\n"
    "n_cols = 2 + len(LAMBDAS) + 1   # imgA + lambdas + random + imgB\n"
    "fig, axes = plt.subplots(N_PAIRS, n_cols,\n"
    "                         figsize=(2.2 * n_cols, 2.4 * N_PAIRS))\n\n"
    "col_titles = (['Image A'] +\n"
    "              [f'λ={l}' for l in LAMBDAS] +\n"
    "              ['λ~Beta(0.2)'] +\n"
    "              ['Image B'])\n\n"
    "for col, title in enumerate(col_titles):\n"
    "    axes[0, col].set_title(title, fontsize=9, fontweight='bold')\n\n"
    "for row, (ai, bi) in enumerate(pairs):\n"
    "    img_a, img_b = imgs[ai], imgs[bi]\n"
    "    lbl_a, lbl_b = CLASS_NAMES[labels[ai]], CLASS_NAMES[labels[bi]]\n\n"
    "    col = 0\n"
    "    # Image A\n"
    "    axes[row, col].imshow(img_a)\n"
    "    axes[row, col].set_xlabel(lbl_a, fontsize=7)\n"
    "    col += 1\n\n"
    "    # Fixed lambda mixes\n"
    "    for lam in LAMBDAS:\n"
    "        mixed = to_display(mixup(img_a, img_b, lam))\n"
    "        axes[row, col].imshow(mixed)\n"
    "        axes[row, col].set_xlabel(f'{lam:.0%} A + {1-lam:.0%} B', fontsize=7)\n"
    "        col += 1\n\n"
    "    # Random lambda from Beta(0.2, 0.2)\n"
    "    lam_rand = float(np.random.beta(0.2, 0.2))\n"
    "    mixed_rand = to_display(mixup(img_a, img_b, lam_rand))\n"
    "    axes[row, col].imshow(mixed_rand)\n"
    "    axes[row, col].set_xlabel(f'{lam_rand:.2f} A + {1-lam_rand:.2f} B', fontsize=7)\n"
    "    col += 1\n\n"
    "    # Image B\n"
    "    axes[row, col].imshow(img_b)\n"
    "    axes[row, col].set_xlabel(lbl_b, fontsize=7)\n\n"
    "for ax in axes.flat:\n"
    "    ax.set_xticks([]); ax.set_yticks([])\n"
    "    for spine in ax.spines.values(): spine.set_visible(False)\n\n"
    "fig.suptitle('Mixup Image Grid  (α=0.2)', fontsize=12, fontweight='bold', y=1.01)\n"
    "plt.tight_layout()\n"
    "plt.savefig('mixup_image_grid.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Saved mixup_image_grid.png')",
    "mv-grid-plot"
))

# ── Plot 3: Alpha sweep ────────────────────────────────────────────────────────
cells.append(md(
    "## 3 — Alpha Sweep\n\n"
    "Same image pair, same λ=0.5, but drawn from different Beta(α, α) distributions. "
    "Since λ is fixed at 0.5 here, the mixed image looks identical — "
    "the alpha affects **how often** extreme vs moderate mixing occurs, not the image itself at λ=0.5. "
    "Below we instead show **random draws** from each alpha to visualise the real effect.",
    "mv-alpha-hdr"
))

cells.append(code(
    "ALPHAS    = [0.1, 0.2, 0.4, 1.0, 2.0, 4.0]\n"
    "N_SAMPLES = 5   # samples per alpha\n\n"
    "# Use fixed pair for comparability\n"
    "ai, bi = pairs[0]\n"
    "img_a, img_b = imgs[ai], imgs[bi]\n"
    "lbl_a, lbl_b = CLASS_NAMES[labels[ai]], CLASS_NAMES[labels[bi]]\n\n"
    "fig, axes = plt.subplots(len(ALPHAS), N_SAMPLES + 2,\n"
    "                         figsize=(2.0 * (N_SAMPLES + 2), 2.2 * len(ALPHAS)))\n\n"
    "for row, alpha in enumerate(ALPHAS):\n"
    "    # First col: Image A\n"
    "    axes[row, 0].imshow(img_a)\n"
    "    axes[row, 0].set_ylabel(f'α={alpha}', fontsize=9, fontweight='bold', rotation=0,\n"
    "                            labelpad=40, va='center')\n"
    "    if row == 0:\n"
    "        axes[row, 0].set_title('Image A', fontsize=8)\n\n"
    "    # Middle cols: random lambda samples\n"
    "    lambdas = np.random.beta(alpha, alpha, N_SAMPLES)\n"
    "    for col, lam in enumerate(lambdas, start=1):\n"
    "        mixed = to_display(mixup(img_a, img_b, lam))\n"
    "        axes[row, col].imshow(mixed)\n"
    "        axes[row, col].set_xlabel(f'λ={lam:.2f}', fontsize=7)\n"
    "        if row == 0:\n"
    "            axes[row, col].set_title(f'Sample {col}', fontsize=8)\n\n"
    "    # Last col: Image B\n"
    "    axes[row, -1].imshow(img_b)\n"
    "    if row == 0:\n"
    "        axes[row, -1].set_title('Image B', fontsize=8)\n\n"
    "for ax in axes.flat:\n"
    "    ax.set_xticks([]); ax.set_yticks([])\n"
    "    for spine in ax.spines.values(): spine.set_visible(False)\n\n"
    "fig.suptitle(\n"
    "    f'Alpha Sweep — {lbl_a} × {lbl_b}\\n'\n"
    "    'Each row: 5 random λ draws from Beta(α, α)',\n"
    "    fontsize=11, fontweight='bold'\n"
    ")\n"
    "plt.tight_layout()\n"
    "plt.savefig('mixup_alpha_sweep.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Saved mixup_alpha_sweep.png')",
    "mv-alpha-plot"
))

# ── Plot 4: Label interpolation diagram ───────────────────────────────────────
cells.append(md(
    "## 4 — Label Interpolation Diagram\n\n"
    "Shows how the soft label changes as λ varies for one example pair.",
    "mv-label-hdr"
))

cells.append(code(
    "ai, bi = pairs[1] if len(pairs) > 1 else pairs[0]\n"
    "img_a, img_b = imgs[ai], imgs[bi]\n"
    "cls_a, cls_b = labels[ai], labels[bi]\n"
    "lbl_a, lbl_b = CLASS_NAMES[cls_a], CLASS_NAMES[cls_b]\n\n"
    "lambdas_sweep = np.linspace(0, 1, 7)\n"
    "n = len(lambdas_sweep)\n\n"
    "fig = plt.figure(figsize=(14, 5))\n"
    "gs  = gridspec.GridSpec(2, n, height_ratios=[3, 1], hspace=0.05)\n\n"
    "for col, lam in enumerate(lambdas_sweep):\n"
    "    ax_img = fig.add_subplot(gs[0, col])\n"
    "    mixed  = to_display(mixup(img_a, img_b, lam))\n"
    "    ax_img.imshow(mixed)\n"
    "    ax_img.set_xticks([]); ax_img.set_yticks([])\n"
    "    for spine in ax_img.spines.values(): spine.set_visible(False)\n"
    "    ax_img.set_title(f'λ={lam:.2f}', fontsize=8)\n\n"
    "    ax_bar = fig.add_subplot(gs[1, col])\n"
    "    ax_bar.bar([0], [lam],     color='#2196F3', width=0.4)\n"
    "    ax_bar.bar([1], [1 - lam], color='#FF5722', width=0.4)\n"
    "    ax_bar.set_ylim(0, 1.1)\n"
    "    ax_bar.set_xticks([0, 1])\n"
    "    ax_bar.set_xticklabels([lbl_a[:5], lbl_b[:5]], fontsize=6, rotation=20)\n"
    "    ax_bar.set_yticks([])\n"
    "    for spine in ax_bar.spines.values(): spine.set_visible(False)\n\n"
    "fig.suptitle(\n"
    "    f'Label interpolation: \"{lbl_a}\" (blue) × \"{lbl_b}\" (orange)',\n"
    "    fontsize=11, fontweight='bold'\n"
    ")\n"
    "plt.savefig('mixup_label_interp.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('Saved mixup_label_interp.png')",
    "mv-label-plot"
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
