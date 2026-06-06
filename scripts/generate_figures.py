"""
DOVE — Generate visualizations comparing experiment streams.
Saves all figures to /data/DOVE/results_comparison/figures/.
Errors in individual figures are caught and logged; script continues.
"""
from __future__ import annotations

import importlib
import logging
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, "/data/DOVE/src")
sys.path.insert(0, "/data/DOVE/scripts")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("/data/generate_figures.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("generate_figures")

FIGS_DIR = Path("/data/DOVE/results_comparison/figures")
FIGS_DIR.mkdir(parents=True, exist_ok=True)

VB100_SPECIES_IDS = {0, 4, 8, 17, 18}  # Acorn WP, Black Phoebe, CA Towhee, RTH, WC Sparrow
SPECIES_NAMES = [
    "Acorn Woodpecker", "American Crow", "American Robin", "Anna's Hummingbird",
    "Black Phoebe", "Brewer's Blackbird", "Bushtit", "California Scrub-Jay",
    "California Towhee", "Chestnut-backed Chickadee", "Cooper's Hawk",
    "Dark-eyed Junco", "House Finch", "Lesser Goldfinch", "Mourning Dove",
    "Northern Mockingbird", "Oak Titmouse", "Red-tailed Hawk",
    "White-crowned Sparrow", "Yellow-rumped Warbler",
]
SHORT_NAMES = [n.split()[0] if n != "Anna's Hummingbird" else "Anna's"
               for n in SPECIES_NAMES]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Using device: %s", device)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Stream comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────
def fig1_stream_comparison():
    log.info("Figure 1: stream_comparison.png")
    entries = [
        ("backbone_only\n(old dataset)",     90.47, "Feature Ablation"),
        ("invariant_only\n(old dataset)",    91.68, "Feature Ablation"),
        ("BB+Inv\n(old dataset)",            92.88, "Feature Ablation"),
        ("HPO best\nBB+Inv (old dataset)",   94.19, "HPO"),
        ("Motion+Inv\nbest",                 87.25, "Motion+Inv"),
        ("Triple best\n(BB+Inv+Motion)",     89.93, "Triple"),
        ("BB+Inv+VB100\nbest",               96.77, "BB+Inv VB100"),
    ]
    labels  = [e[0] for e in entries]
    values  = [e[1] for e in entries]
    groups  = [e[2] for e in entries]

    palette = {
        "Feature Ablation": "#4C72B0",
        "HPO":              "#DD8452",
        "Motion+Inv":       "#55A868",
        "Triple":           "#C44E52",
        "BB+Inv VB100":     "#8172B3",
    }
    colors = [palette[g] for g in groups]
    best_val = max(values)

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.6)
    ax.axvline(best_val, color="red", linestyle="--", linewidth=1.5, label=f"Best: {best_val}%")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", ha="left", fontsize=9.5, fontweight="bold")

    ax.set_xlabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("DOVE: Test Accuracy Across All Stream Combinations", fontsize=13, fontweight="bold")
    ax.set_xlim(84, 98.5)
    ax.invert_yaxis()

    legend_handles = [mpatches.Patch(color=c, label=g) for g, c in palette.items()]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGS_DIR / "stream_comparison.png", dpi=150)
    plt.close(fig)
    log.info("  saved stream_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Backbone comparison (best per backbone from vb100)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_backbone_comparison():
    log.info("Figure 2: backbone_comparison.png")
    df = pd.read_csv("/data/DOVE/results_vb100/tables/experiment_results.csv")
    # only neural configs
    neural = df[df["head"].isin(["mlp", "linear"])].copy()
    best_per_bb = (neural.groupby("backbone")["test_accuracy"]
                   .max()
                   .reset_index()
                   .sort_values("test_accuracy", ascending=False))
    best_per_bb["test_pct"] = best_per_bb["test_accuracy"] * 100

    backbone_order = ["efficientnet_b3", "swin_t", "vgg19", "mobilenet"]
    bb_labels = {"efficientnet_b3": "EfficientNet-B3", "swin_t": "Swin-T",
                 "vgg19": "VGG-19", "mobilenet": "MobileNet"}
    colors = ["#8172B3", "#4C72B0", "#55A868", "#DD8452"]

    fig, ax = plt.subplots(figsize=(8, 5))
    vals = [best_per_bb.loc[best_per_bb["backbone"] == bb, "test_pct"].values[0]
            for bb in backbone_order]
    bars = ax.bar([bb_labels[b] for b in backbone_order], vals,
                  color=colors, edgecolor="white", width=0.55)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold")

    ax.set_ylabel("Best Test Accuracy (%)", fontsize=12)
    ax.set_title("Best Test Accuracy per Backbone (BB+Inv, VB100 dataset)", fontsize=13, fontweight="bold")
    ax.set_ylim(93, 98)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "backbone_comparison.png", dpi=150)
    plt.close(fig)
    log.info("  saved backbone_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Fusion × Head heatmap
# ─────────────────────────────────────────────────────────────────────────────
def fig3_fusion_head_heatmap():
    log.info("Figure 3: fusion_head_heatmap.png")
    import seaborn as sns

    df = pd.read_csv("/data/DOVE/results_vb100/tables/experiment_results.csv")
    neural = df[df["head"].isin(["mlp", "linear"])].copy()
    neural["combo"] = neural["fusion"] + "_" + neural["head"]
    neural["test_pct"] = neural["test_accuracy"] * 100

    backbone_order = ["efficientnet_b3", "swin_t", "mobilenet", "vgg19"]
    combo_order    = ["cross_attention_mlp", "cross_attention_linear",
                      "concat_mlp", "concat_linear"]

    pivot = neural.pivot_table(index="backbone", columns="combo",
                               values="test_pct", aggfunc="mean")
    pivot = pivot.reindex(index=backbone_order, columns=combo_order)

    bb_labels = {"efficientnet_b3": "EfficientNet-B3", "swin_t": "Swin-T",
                 "mobilenet": "MobileNet", "vgg19": "VGG-19"}
    combo_labels = {"cross_attention_mlp": "CrossAttn\n+MLP",
                    "cross_attention_linear": "CrossAttn\n+Linear",
                    "concat_mlp": "Concat\n+MLP",
                    "concat_linear": "Concat\n+Linear"}

    pivot.index   = [bb_labels[b] for b in pivot.index]
    pivot.columns = [combo_labels[c] for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, vmin=95, vmax=97,
                annot_kws={"size": 11, "weight": "bold"})
    ax.set_title("Test Accuracy (%) — Backbone × Fusion+Head (BB+Inv, VB100)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Fusion + Head", fontsize=11)
    ax.set_ylabel("Backbone", fontsize=11)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "fusion_head_heatmap.png", dpi=150)
    plt.close(fig)
    log.info("  saved fusion_head_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# Shared: load BB+Inv model
# ─────────────────────────────────────────────────────────────────────────────
def load_bbinv_model():
    from features.invariant.pipeline import InvariantFeaturePipeline
    from fusion.cross_attention import CrossAttentionFusion
    from heads.mlp_head import MLPHead

    class _NeuralModel(nn.Module):
        def __init__(self, backbone, inv_pipeline, fusion, head):
            super().__init__()
            self.backbone     = backbone
            self.inv_pipeline = inv_pipeline
            self.fusion       = fusion
            self.head         = head

        def forward(self, images):
            bb_feat = self.backbone(images)
            if bb_feat.dim() > 2:
                bb_feat = bb_feat.mean(dim=[-2, -1])
            inv_feat = self.inv_pipeline.extract(images)
            fused    = self.fusion(bb_feat, inv_feat)
            return self.head(fused)

    bb_mod      = importlib.import_module("backbones.efficientnet_b3")
    backbone_cls = getattr(bb_mod, [x for x in dir(bb_mod) if "Backbone" in x][0])
    backbone     = backbone_cls().to(device)
    fusion       = CrossAttentionFusion(motion_dim=backbone.feature_dim).to(device)
    head         = MLPHead(in_dim=512, num_classes=20, hidden_dim=256, dropout=0.3).to(device)
    model = _NeuralModel(backbone, InvariantFeaturePipeline().to(device), fusion, head).to(device)
    ckpt = "/data/DOVE/results_vb100/checkpoints/efficientnet_b3_cross_attention_mlp_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    log.info("  Loaded BB+Inv model from %s", ckpt)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Shared: load Triple model
# ─────────────────────────────────────────────────────────────────────────────
def load_triple_model():
    from run_experiments_triple import (
        TripleCrossAttentionFusion, TripleStreamModel
    )
    from features.invariant.pipeline import InvariantFeaturePipeline

    bb_mod       = importlib.import_module("backbones.swin_t")
    backbone_cls = getattr(bb_mod, [x for x in dir(bb_mod) if "Backbone" in x][0])
    backbone     = backbone_cls().to(device)
    inv_pipeline = InvariantFeaturePipeline().to(device)
    fusion       = TripleCrossAttentionFusion(backbone_dim=backbone.feature_dim).to(device)
    # triple checkpoint stores head as plain Sequential (head.0, head.3, etc.)
    head         = nn.Sequential(
        nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.316), nn.Linear(256, 20)
    ).to(device)  # matches triple_swin_t checkpoint keys
    model = TripleStreamModel(backbone, inv_pipeline, fusion, head).to(device)
    ckpt = "/data/DOVE/results_triple/checkpoints/triple_swin_t_cross_attention_mlp_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    log.info("  Loaded Triple model from %s", ckpt)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Shared: build test loader and motion lookup
# ─────────────────────────────────────────────────────────────────────────────
def build_test_loader():
    from data.loader import DOVEDataset
    from data.augment import get_image_transform
    transform = get_image_transform(train=False)
    ds = DOVEDataset("/data/DOVE/data/splits/test.csv", transform=transform)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    return loader


def build_stem_to_motion():
    mpath = Path("/data/DOVE/data/motion_features.parquet")
    if not mpath.exists():
        log.warning("motion_features.parquet not found; using zero motion")
        return {}
    df = pd.read_parquet(mpath)
    stem_col = "stem" if "stem" in df.columns else df.columns[0]
    feat_cols = [c for c in df.columns if c != stem_col]
    mapping = {row[stem_col]: torch.tensor(row[feat_cols].values.astype(np.float32))
               for _, row in df.iterrows()}
    log.info("  Motion lookup: %d entries", len(mapping))
    return mapping


def run_inference(model, loader, stem_to_motion=None, is_triple=False):
    """Returns (all_preds, all_labels) as numpy arrays."""
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            if is_triple:
                fps = batch["filepath"]
                mot = torch.stack([
                    stem_to_motion.get(Path(fp).stem,
                                       torch.zeros(139))
                    for fp in fps
                ]).to(device)
                logits = model(images, mot)
            else:
                logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Motion impact per species
# ─────────────────────────────────────────────────────────────────────────────
def fig4_motion_impact_per_species():
    log.info("Figure 4: motion_impact_per_species.png")
    loader           = build_test_loader()
    stem_to_motion   = build_stem_to_motion()

    log.info("  Loading BB+Inv model…")
    bbinv_model  = load_bbinv_model()
    log.info("  Running BB+Inv inference…")
    preds_bbinv, labels = run_inference(bbinv_model, loader, is_triple=False)
    del bbinv_model

    log.info("  Loading Triple model…")
    triple_model = load_triple_model()
    log.info("  Running Triple inference…")
    preds_triple, _ = run_inference(triple_model, loader, stem_to_motion=stem_to_motion, is_triple=True)
    del triple_model

    n_classes = 20
    acc_bbinv  = np.array([np.mean(preds_bbinv[labels == c]  == c) * 100
                            for c in range(n_classes)])
    acc_triple = np.array([np.mean(preds_triple[labels == c] == c) * 100
                            for c in range(n_classes)])

    x       = np.arange(n_classes)
    width   = 0.38
    colors_bbinv  = ["#C44E52" if i in VB100_SPECIES_IDS else "#4C72B0" for i in range(n_classes)]
    colors_triple = ["#FF8C00"  if i in VB100_SPECIES_IDS else "#55A868" for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - width/2, acc_bbinv,  width, color=colors_bbinv,  label="BB+Inv (EfficientNet-B3)",  edgecolor="white")
    ax.bar(x + width/2, acc_triple, width, color=colors_triple, label="Triple (Swin-T+Motion)",     edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_NAMES, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("Per-species Accuracy (%)", fontsize=11)
    ax.set_title("Per-Species Accuracy: BB+Inv vs Triple Model\n"
                 "(Red/Orange bars = VB100 species with real motion data)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)

    legend_handles = [
        mpatches.Patch(color="#4C72B0", label="BB+Inv — iNat species"),
        mpatches.Patch(color="#C44E52", label="BB+Inv — VB100 species"),
        mpatches.Patch(color="#55A868", label="Triple — iNat species"),
        mpatches.Patch(color="#FF8C00", label="Triple — VB100 species"),
    ]
    ax.legend(handles=legend_handles, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "motion_impact_per_species.png", dpi=150)
    plt.close(fig)
    log.info("  saved motion_impact_per_species.png")
    return labels, preds_bbinv


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────
def fig5_confusion_matrix(labels=None, preds=None):
    log.info("Figure 5: confusion_matrix.png")
    import seaborn as sns

    if labels is None or preds is None:
        loader = build_test_loader()
        model  = load_bbinv_model()
        preds, labels = run_inference(model, loader, is_triple=False)
        del model

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds, labels=list(range(20)))
    # row-normalize (recall)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
                linewidths=0.3, ax=ax, vmin=0, vmax=1,
                annot_kws={"size": 7})
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Confusion Matrix — BB+Inv (EfficientNet-B3 + CrossAttn, VB100)\n(Row-normalized: Recall)",
                 fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    log.info("  saved confusion_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Score-CAM grid
# ─────────────────────────────────────────────────────────────────────────────
def fig6_scorecam_grid():
    log.info("Figure 6: scorecam_grid.png")
    import torch.nn.functional as F
    from PIL import Image as PILImage
    from data.loader import DOVEDataset
    from data.augment import get_image_transform

    # species pairs that are visually interesting / confusable
    TARGET_SPECIES = [10, 17, 3, 12, 6, 9]  # Cooper's Hawk, Red-tailed Hawk,
                                             # Anna's Hummingbird, House Finch, Bushtit, CB Chickadee

    # Build BB+Inv model (with hook on last conv layer)
    from features.invariant.pipeline import InvariantFeaturePipeline
    from fusion.cross_attention import CrossAttentionFusion

    bb_mod       = importlib.import_module("backbones.efficientnet_b3")
    backbone_cls = getattr(bb_mod, [x for x in dir(bb_mod) if "Backbone" in x][0])
    backbone     = backbone_cls().to(device)

    # Find the EfficientNet feature extractor
    # EfficientNet-B3 backbone wraps torchvision's efficientnet_b3
    # We hook backbone.model.features[-1] or similar
    eff_net = None
    for name, mod in backbone.named_modules():
        if "features" in name.lower():
            eff_net = mod
    # Try to find the actual torchvision EfficientNet features block
    feature_extractor = None
    for attr in ["model", "backbone", "net", "features"]:
        if hasattr(backbone, attr):
            obj = getattr(backbone, attr)
            if hasattr(obj, "features"):
                feature_extractor = obj.features
                break
            elif callable(obj):
                feature_extractor = obj
                break
    if feature_extractor is None:
        # Walk children to find a Sequential with 9 items (EfficientNet features)
        for name, mod in backbone.named_children():
            if isinstance(mod, nn.Sequential) and len(list(mod.children())) >= 5:
                feature_extractor = mod
                break
    if feature_extractor is None:
        log.warning("  Could not identify EfficientNet feature layers; trying backbone directly")
        feature_extractor = backbone

    log.info("  Feature extractor type: %s, children: %d",
             type(feature_extractor).__name__,
             len(list(feature_extractor.children())))

    # Hook the last child of the feature extractor
    activations = {}
    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    last_layer = list(feature_extractor.children())[-1]
    hook_handle = last_layer.register_forward_hook(save_activation("last_conv"))

    from heads.mlp_head import MLPHead as _MLPHead
    fusion = CrossAttentionFusion(motion_dim=backbone.feature_dim).to(device)
    head   = _MLPHead(in_dim=512, num_classes=20, hidden_dim=256, dropout=0.3).to(device)

    class _NeuralModel(nn.Module):
        def __init__(self, backbone, inv_pipeline, fusion, head):
            super().__init__()
            self.backbone     = backbone
            self.inv_pipeline = inv_pipeline
            self.fusion       = fusion
            self.head         = head
        def forward(self, images):
            bb_feat = self.backbone(images)
            if bb_feat.dim() > 2:
                bb_feat = bb_feat.mean(dim=[-2, -1])
            inv_feat = self.inv_pipeline.extract(images)
            fused    = self.fusion(bb_feat, inv_feat)
            return self.head(fused)

    model = _NeuralModel(backbone, InvariantFeaturePipeline().to(device), fusion, head).to(device)
    ckpt  = "/data/DOVE/results_vb100/checkpoints/efficientnet_b3_cross_attention_mlp_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    transform_val = get_image_transform(train=False)
    ds = DOVEDataset("/data/DOVE/data/splits/test.csv", transform=transform_val)
    df = ds.df.copy()

    # Pick one test image per target species
    samples = []
    for sp_id in TARGET_SPECIES:
        rows = df[df["species_id"] == sp_id]
        if len(rows) == 0:
            log.warning("  No test samples for species %d", sp_id)
            continue
        row = rows.iloc[0]
        fpath = str(row["filepath"])
        samples.append((sp_id, fpath))

    def score_cam(img_tensor, target_class):
        """img_tensor: (1, 3, 224, 224)"""
        img_tensor = img_tensor.to(device)
        activations.clear()

        with torch.no_grad():
            _ = model(img_tensor)  # populate activations["last_conv"]

        if "last_conv" not in activations:
            log.warning("    activation hook did not fire")
            return None

        act = activations["last_conv"]  # (1, C, h, w)
        if act.dim() == 2:
            log.warning("    activation is 2D, skipping Score-CAM")
            return None

        C = act.shape[1]
        scores = []
        for c in range(C):
            channel = act[0, c]  # (h, w)
            # Upscale to 224×224
            channel_up = F.interpolate(
                channel.unsqueeze(0).unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
            )[0, 0]
            # Normalize to [0, 1]
            mn, mx = channel_up.min(), channel_up.max()
            if mx - mn < 1e-8:
                mask = torch.zeros_like(channel_up)
            else:
                mask = (channel_up - mn) / (mx - mn)
            # Masked input
            masked = img_tensor * mask.unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logit = model(masked)
            score = torch.softmax(logit, dim=1)[0, target_class].item()
            scores.append(score)

        scores_t = torch.tensor(scores, device=device)  # (C,)
        act_map   = act[0]  # (C, h, w)
        weighted  = (scores_t[:, None, None] * act_map).sum(dim=0)
        cam       = F.relu(weighted)
        cam_up    = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()
        mn, mx = cam_up.min(), cam_up.max()
        if mx - mn > 1e-8:
            cam_up = (cam_up - mn) / (mx - mn)
        return cam_up

    # Imagenet mean/std for de-normalization
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    n_samples = len(samples)
    fig, axes = plt.subplots(n_samples, 2, figsize=(8, 3 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (sp_id, fpath) in enumerate(samples):
        try:
            img_pil = PILImage.open(fpath).convert("RGB").resize((224, 224))
        except Exception as e:
            log.warning("  Could not open %s: %s", fpath, e)
            continue

        img_tensor = transform_val(img_pil).unsqueeze(0)
        with torch.no_grad():
            logits = model(img_tensor.to(device))
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_class = int(np.argmax(probs))
        confidence = probs[pred_class] * 100

        cam = score_cam(img_tensor, target_class=sp_id)

        # De-normalize for display
        img_np = img_tensor[0].permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * std + mean, 0, 1)

        # Original image
        axes[row_idx, 0].imshow(img_np)
        axes[row_idx, 0].axis("off")
        axes[row_idx, 0].set_title(f"{SPECIES_NAMES[sp_id]}\n(original)", fontsize=8)

        # Score-CAM overlay
        axes[row_idx, 1].imshow(img_np)
        if cam is not None:
            axes[row_idx, 1].imshow(cam, alpha=0.5, cmap="jet")
        pred_name = SPECIES_NAMES[pred_class]
        correct = "✓" if pred_class == sp_id else "✗"
        axes[row_idx, 1].axis("off")
        axes[row_idx, 1].set_title(
            f"Score-CAM | Pred: {pred_name} {correct} ({confidence:.1f}%)", fontsize=8
        )

    hook_handle.remove()
    fig.suptitle("Score-CAM: Best BB+Inv Model (EfficientNet-B3 + CrossAttn)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "scorecam_grid.png", dpi=150)
    plt.close(fig)
    log.info("  saved scorecam_grid.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run_all():
    tasks = [
        ("stream_comparison",          fig1_stream_comparison),
        ("backbone_comparison",         fig2_backbone_comparison),
        ("fusion_head_heatmap",         fig3_fusion_head_heatmap),
        ("motion_impact_per_species",   None),
        ("confusion_matrix",            None),
        ("scorecam_grid",               fig6_scorecam_grid),
    ]

    # Figures 4 + 5 share inference, so run together to save time
    labels_cache = None
    preds_cache  = None

    for name, fn in tasks:
        if name == "motion_impact_per_species":
            try:
                labels_cache, preds_cache = fig4_motion_impact_per_species()
            except Exception:
                log.error("FAILED %s:\n%s", name, traceback.format_exc())
        elif name == "confusion_matrix":
            try:
                fig5_confusion_matrix(labels=labels_cache, preds=preds_cache)
            except Exception:
                log.error("FAILED %s:\n%s", name, traceback.format_exc())
        else:
            try:
                fn()
            except Exception:
                log.error("FAILED %s:\n%s", name, traceback.format_exc())

    log.info("All figures attempted. Saved to %s", FIGS_DIR)


if __name__ == "__main__":
    run_all()
