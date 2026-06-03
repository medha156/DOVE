"""
DOVE — Stratified train/val/test split generation.

Splits are performed at the video/image level (not frame level).
Stratification is by a combined (species_id, modality) key.

Data sources come from the CHIRP repo (https://github.com/medha156/CHIRP):
  - iNaturalist images: data/inaturalist/index.csv  (columns: path, label, species, ...)
  - VB100 filtered frames: data/processed/vb100_frames/index.csv
    These frames have already been quality-filtered by pixel std (>=10) and
    Laplacian variance (>=5) — blank/blurred frames are excluded.
    VB100 frames are stored as JPEGs, split is done at the video_src level
    to prevent data leakage across frame siblings of the same clip.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

SPECIES_NAMES = [
    "Acorn Woodpecker", "American Crow", "American Robin", "Anna's Hummingbird",
    "Black Phoebe", "Brewer's Blackbird", "Bushtit", "California Scrub-Jay",
    "California Towhee", "Chestnut-backed Chickadee", "Cooper's Hawk",
    "Dark-eyed Junco", "House Finch", "Lesser Goldfinch", "Mourning Dove",
    "Northern Mockingbird", "Oak Titmouse", "Red-tailed Hawk",
    "White-crowned Sparrow", "Yellow-rumped Warbler",
]
SPECIES_TO_ID = {name: i for i, name in enumerate(SPECIES_NAMES)}

# CHIRP uses lowercase slugs for species directories
SLUG_TO_ID = {name.lower().replace("'", "").replace(" ", "_"): i
              for i, name in enumerate(SPECIES_NAMES)}
# also map integer label directly
_LABEL_TO_ID = {i: i for i in range(len(SPECIES_NAMES))}


def _resolve_species_id(row: pd.Series) -> int:
    """Resolve species_id from CHIRP index row (label col is int 0-19)."""
    if "label" in row.index:
        return int(row["label"])
    if "species_id" in row.index:
        return int(row["species_id"])
    if "species" in row.index:
        name = str(row["species"])
        sid = SPECIES_TO_ID.get(name, SLUG_TO_ID.get(name.lower().replace(" ", "_"), -1))
        return sid
    return -1


def _resolve_species_name(row: pd.Series) -> str:
    sid = _resolve_species_id(row)
    if 0 <= sid < len(SPECIES_NAMES):
        return SPECIES_NAMES[sid]
    return row.get("species", "unknown")


def build_manifest_from_chirp(
    inat_index: str | Path,
    vb100_index: str | Path,
    data_root: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Build a unified manifest from CHIRP index CSVs.

    CHIRP iNat index columns: path, label, species, source, modality, license
    CHIRP VB100 index columns: path, label, species, source, modality, license,
                                video_src, frame_idx

    Returns DataFrame with columns:
        filepath, species_id, species_name, modality, video_src
    where video_src is the originating clip path (for leak-free video splits)
    and modality is 'image' for iNat and 'video_frame' for VB100 frames.
    """
    records: list[dict] = []
    data_root = Path(data_root) if data_root else None

    for index_path, modality_tag in [
        (inat_index, "image"),
        (vb100_index, "video_frame"),
    ]:
        index_path = Path(index_path)
        if not index_path.exists():
            logger.warning("Index not found: %s — skipping", index_path)
            continue

        df = pd.read_csv(index_path)
        logger.info("Loaded %d rows from %s", len(df), index_path)

        path_col = "path" if "path" in df.columns else "filepath"

        for _, row in df.iterrows():
            sid = _resolve_species_id(row)
            if sid < 0:
                continue
            fpath = str(row[path_col])
            if data_root and not Path(fpath).is_absolute():
                fpath = str(data_root / fpath)
            records.append({
                "filepath": fpath,
                "species_id": sid,
                "species_name": SPECIES_NAMES[sid],
                "modality": modality_tag,
                "video_src": str(row.get("video_src", "")),
            })

    if not records:
        logger.warning("No data found in CHIRP indices")
        return pd.DataFrame(
            columns=["filepath", "species_id", "species_name", "modality", "video_src"]
        )
    return pd.DataFrame(records)


def build_manifest(
    inat_root: str | Path,
    vb100_frames_root: str | Path,
) -> pd.DataFrame:
    """
    Backward-compatible scanner: reads index.csv from each root if present,
    otherwise falls back to scanning the directory.
    Prefer build_manifest_from_chirp when CHIRP index CSVs are available.
    """
    inat_root = Path(inat_root)
    vb100_frames_root = Path(vb100_frames_root)

    inat_index = inat_root / "index.csv"
    vb100_index = vb100_frames_root / "index.csv"

    if inat_index.exists() or vb100_index.exists():
        logger.info("Found CHIRP index CSVs — using build_manifest_from_chirp")
        return build_manifest_from_chirp(
            inat_index, vb100_index, data_root=None
        )

    # Fallback: directory scan
    records: list[dict] = []
    for root, modality_tag, extensions in [
        (inat_root, "image", {".jpg", ".jpeg", ".png"}),
        (vb100_frames_root, "video_frame", {".jpg", ".jpeg", ".png"}),
    ]:
        if not root.exists():
            logger.warning("Directory %s does not exist — skipping", root)
            continue
        for species_dir in sorted(root.iterdir()):
            if not species_dir.is_dir():
                continue
            sid = SPECIES_TO_ID.get(species_dir.name,
                  SLUG_TO_ID.get(species_dir.name, -1))
            if sid < 0:
                logger.warning("Unknown species directory: %s", species_dir.name)
                continue
            for fpath in sorted(species_dir.iterdir()):
                if fpath.suffix.lower() in extensions:
                    records.append({
                        "filepath": str(fpath),
                        "species_id": sid,
                        "species_name": SPECIES_NAMES[sid],
                        "modality": modality_tag,
                        "video_src": "",
                    })
    if not records:
        logger.warning("No data found in %s or %s", inat_root, vb100_frames_root)
        return pd.DataFrame(
            columns=["filepath", "species_id", "species_name", "modality", "video_src"]
        )
    return pd.DataFrame(records)


def generate_splits(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Add a 'split' column with values 'train'/'val'/'test'.
    Stratified by (species_id, modality).

    For VB100 video frames (modality='video_frame'), splitting is done at
    the clip level (video_src) to prevent frame siblings of the same clip
    appearing in both train and val/test sets.
    """
    df = df.copy()

    # ── 1. Split image rows directly ──────────────────────────────────────
    img_df = df[df["modality"] == "image"].copy()
    vid_df = df[df["modality"] == "video_frame"].copy()

    def _stratified_split(sub_df: pd.DataFrame, strat_col: str) -> pd.DataFrame:
        sub_df = sub_df.copy()
        strat = sub_df[strat_col].values
        n = len(sub_df)
        indices = np.arange(n)

        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        trainval_idx, test_idx = next(sss_test.split(indices, strat))

        val_frac_adj = val_frac / (1.0 - test_frac)
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_adj, random_state=seed)
        train_sub, val_sub = next(sss_val.split(trainval_idx, strat[trainval_idx]))

        sub_df["split"] = "train"
        sub_df.iloc[trainval_idx[val_sub], sub_df.columns.get_loc("split")] = "val"
        sub_df.iloc[test_idx, sub_df.columns.get_loc("split")] = "test"
        return sub_df

    if len(img_df) > 0:
        img_df["strat_key"] = img_df["species_id"].astype(str)
        img_df = _stratified_split(img_df, "strat_key")
        img_df = img_df.drop(columns=["strat_key"])

    # ── 2. Split VB100 at clip level, then broadcast to frames ───────────
    if len(vid_df) > 0:
        has_src = vid_df["video_src"].notna() & (vid_df["video_src"] != "")
        if has_src.any():
            # Build a clip-level manifest
            clip_df = (
                vid_df[has_src]
                .groupby("video_src", as_index=False)
                .agg(species_id=("species_id", "first"))
            )
            clip_df["strat_key"] = clip_df["species_id"].astype(str)
            clip_df = _stratified_split(clip_df, "strat_key")
            clip_split = clip_df.set_index("video_src")["split"].to_dict()

            vid_df["split"] = vid_df["video_src"].map(clip_split).fillna("train")
            # rows without video_src fall back to frame-level split
            no_src = vid_df[~has_src].copy()
            if len(no_src) > 0:
                no_src["strat_key"] = no_src["species_id"].astype(str)
                no_src = _stratified_split(no_src, "strat_key")
                no_src = no_src.drop(columns=["strat_key"])
                vid_df = pd.concat([vid_df[has_src], no_src], ignore_index=True)
        else:
            vid_df["strat_key"] = vid_df["species_id"].astype(str)
            vid_df = _stratified_split(vid_df, "strat_key")
            vid_df = vid_df.drop(columns=["strat_key"])

    return pd.concat([img_df, vid_df], ignore_index=True)


def compute_class_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Inverse-frequency weights: w_c = N_total / (N_classes * N_c).
    Returns array of length num_classes.
    """
    train_df = df[df["split"] == "train"]
    n_classes = df["species_id"].nunique()
    n_total = len(train_df)
    weights = np.zeros(n_classes, dtype=np.float32)
    for c in range(n_classes):
        n_c = (train_df["species_id"] == c).sum()
        weights[c] = n_total / (n_classes * n_c) if n_c > 0 else 0.0
    return weights


def save_splits(
    df: pd.DataFrame,
    splits_dir: str | Path,
    class_weights: Optional[np.ndarray] = None,
) -> None:
    """Write train/val/test CSVs and optionally class_weights.npy."""
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        subset = df[df["split"] == split].drop(columns=["split"])
        subset.to_csv(splits_dir / f"{split}.csv", index=False)
        logger.info("Saved %d rows to %s/%s.csv", len(subset), splits_dir, split)
    if class_weights is not None:
        np.save(splits_dir / "class_weights.npy", class_weights)
        logger.info("Saved class weights to %s/class_weights.npy", splits_dir)


if __name__ == "__main__":
    import random
    import tempfile

    logging.basicConfig(level=logging.INFO)
    random.seed(42)
    np.random.seed(42)

    # Build a synthetic manifest
    rng = np.random.default_rng(42)
    rows = []
    for sid, sname in enumerate(SPECIES_NAMES):
        for mod, n in [("image", 30), ("video", 20)]:
            for i in range(n):
                rows.append(
                    {
                        "filepath": f"fake/{sname}/{mod}_{i}.{'jpg' if mod=='image' else 'mp4'}",
                        "species_id": sid,
                        "species_name": sname,
                        "modality": mod,
                    }
                )
    df = pd.DataFrame(rows)
    df = generate_splits(df)
    weights = compute_class_weights(df)

    print("Split counts:", df["split"].value_counts().to_dict())
    print("Class weights shape:", weights.shape)
    print("Class weights (first 5):", weights[:5])

    with tempfile.TemporaryDirectory() as tmp:
        save_splits(df, tmp, weights)
        print("CSVs written to", tmp)
        print("train.csv head:")
        print(pd.read_csv(f"{tmp}/train.csv").head(3))
