import os, io, sys, requests, PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from urllib.error import HTTPError
import math
import random

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np, os, json
from pathlib import Path

if os.getcwd() in sys.path:
    sys.path.remove(os.getcwd())
elif '' in sys.path:
    sys.path.remove('')

# from geotessera import GeoTessera
import ee

# AlphaEarth stores "no data" as the float32 minimum, -3.4028235e+38. That value is
# finite, so isinf/isnan never catch it. Real embedding values sit within about
# -0.45 to 0.45, so anything below this threshold is unambiguously a no-data marker.
# Compared as a threshold rather than for equality, because exact float comparison
# is fragile.
NODATA_THRESHOLD = -1e30


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────
class BiomassDataset(Dataset):
    """
    Streams (patch_embedding, patch_target) pairs from pre-saved .npy tiles.

    Directory layout expected:
        data_dir/
            embeddings/       <name>_x.npy       (H, W, 128)[optional]
            ae_embeddings/    <name>_ae.npy       (H, W, C_ae)   
            targets/          <name>_y.npy        (H, W)
            norm_stats.json                        GeoTessera stats[optional]
            norm_stats_ae.json                     AlphaEarth stats 

    Args:
        data_dir    : root folder described above
        patch_size  : square patch side length in pixels
        split       : "train" | "val" | "test"
        split_file  : optional path to a JSON manifest with explicit splits
                      e.g. {"train": ["amazon_2020", ...], "val": [...], "test": [...]}
        use_ae      : whether to load AlphaEarth embeddings (or Tessera)
        augment     : random rot90 + horizontal flip (training only, optional)
    """

    def __init__(self, data_dir, patch_size=64, split="train", split_ratio=(0.7, 0.15, 0.15), use_ae=False, augment=False, seed=42):
        # self.patch_size = patch_size ; self.use_ae = use_ae ; self.augment = augment
        data_dir = Path(data_dir)  # emb_dir  = data_dir / "embeddings"
        ae_dir = data_dir / "ae_embeddings"
        y_dir = data_dir / "targets"

        all_names = [f.stem.replace("_ae", "") for f in ae_dir.glob("*_ae.npy")]

        # Split via _default_split, which uses its own private random generator.
        # The previous version shuffled with the global `random` module here, which
        # meant the ordering depended on how many random numbers anything else had
        # already drawn. compute_normalization_stats() draws 20 of them between the
        # train and val builds, so the two ended up with different orderings and
        # their slices overlapped: 28 of 45 val tiles were also training tiles.
        # Ordering now depends only on `seed`.
        names = self._default_split(all_names, split, split_ratio=split_ratio, seed=seed)

        self.samples = []

        for name in names:
            ae_path = ae_dir / f"{name}_ae.npy"
            y_path  = y_dir / f"{name}_y.npy"

            if ae_path.exists() and y_path.exists():
                self.samples.append((ae_path, y_path, name))

        print(f"{split}: {len(self.samples)} tiles")

        # ae_stats_path = data_dir / "norm_stats_ae.json"
        # if ae_stats_path.exists():
        #     with open(ae_stats_path) as f:
        #         ae_stats = json.load(f)
        #     self.ae_mean = np.array(ae_stats["mean"], dtype=np.float32)
        #     self.ae_std  = np.array(ae_stats["std"],  dtype=np.float32)
        #     print(f"Loaded AE normalization stats → mean shape {self.ae_mean}, std shape {self.ae_std}")
        # else:
        #     if use_ae:
        #         print(f"Warning: {ae_stats_path} not found — skipping AE normalization.")
        #     self.ae_mean = 0 ; self.ae_std  = 1

        # # ── Split manifest ────────────────────────────────────────────────────
        # print("Splitting the current data")
        # if split_file and os.path.exists(split_file):
        #     with open(split_file) as f:
        #         manifest = json.load(f)
        #     if split not in manifest:
        #         raise KeyError(f"Split '{split}' not found in {split_file}")
        #     names = manifest[split]
        # else:
        #     all_names = [f.stem.replace("_ae", "") for f in ae_dir.glob("*_ae.npy")]
        #     if not all_names:
        #         raise FileNotFoundError(f"No *_ae.npy files found in {ae_dir}")
        #     names = self._default_split(all_names, split)

        # # ── Memory-map tiles (no RAM cost until sliced) ───────────────────────
        # self.tiles = []
        # for name in names:
        #     #emb_path = emb_dir / f"{name}_x.npy"
        #     ae_path  = ae_dir / f"{name}_ae.npy"
        #     tgt_path = tgt_dir / f"{name}_y.npy"

        #     if not ae_path.exists() or not tgt_path.exists():
        #         print(f"Warning: missing files for '{name}', skipping.")
        #         continue

        #     ae = np.load(ae_path, mmap_mode='r', )   # (H, W, 128)
        #     tgt = np.load(tgt_path, mmap_mode='r')   # (H, W)
        #     # emb = np.load(emb_path, mmap_mode='r')   # (H, W, C_ae) or (H, W, 128)

        #     # if not use_ae:
        #     #     emb = np.load(emb_path, mmap_mode='r')
               

        #     # H, W, _ = tgt.shape
        #     H, W = tgt.shape[:2]
        #     self.tiles.append({
        #         "name": name,
        #         "ae":  ae,
        #         "tgt":  tgt,
        #         # "emb":  emb if not use_ae else None,  # only load Tessera if AE not used
        #         "H":    H,
        #         "W":    W,
        #     })

        # if not self.tiles:
        #     raise RuntimeError("No valid tiles loaded — check your data directory.")

        # # ── Flat patch index: (tile_idx, row_start, col_start) ───────────────
        # self.index = []
        # for ti, tile in enumerate(self.tiles):
        #     H, W, P = tile["H"], tile["W"], patch_size
        #     for r in range(0, H - P, P):
        #         for c in range(0, W - P, P):
        #             self.index.append((ti, r, c))

        # print(f"BiomassDataset [{split}]: {len(self.tiles)} tiles, "
        #       f"{len(self.index)} patches of size {patch_size}x{patch_size}")

    # ── basic methods ─────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        ae_path, y_path, name = self.samples[idx]

        x = np.load(ae_path)   # (H, W, C)
        y = np.load(y_path)    # (H, W) or (H, W, 1)

        if y.ndim == 2:
            y = y[..., None]

        # Flag no-data pixels (see NODATA_THRESHOLD above). A pixel is unusable if
        # ANY of its 64 channels carries the marker. Those pixels are zeroed in the
        # input and flagged in `valid`, so the loss can skip them: zeroing alone
        # would still leave the model graded on places where there is nothing to
        # predict from. About 8-9% of pixels are affected.
        bad = (x < NODATA_THRESHOLD).any(axis=-1)      # (H, W) True where no data
        x = np.where(x < NODATA_THRESHOLD, 0.0, x)
        valid = (~bad).astype(np.float32)              # 1.0 = usable, 0.0 = ignore

        # to tensor
        x = torch.from_numpy(x).float() ; y = torch.from_numpy(y).float()
        valid = torch.from_numpy(valid)

        return x, y, valid, name

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _augment(self, x, y): # verify x [H,W,C] ?
        """Random rot90 + horizontal flip — both applied identically to x and y."""
        k = np.random.randint(0, 4)
        x = np.rot90(x, k).copy() ; y = np.rot90(y, k).copy()
        if np.random.rand() > 0.5:
            x = np.fliplr(x).copy() ; y = np.fliplr(y).copy()
        return x, y

    @staticmethod
    def _default_split(names, split, split_ratio=(0.7, 0.15, 0.15), seed=42):
        """Reproducible train/val/test split.

        Uses its own private generator (np.random.default_rng) rather than the
        global `random` module, so the ordering depends only on `seed` and cannot
        be disturbed by anything else that draws random numbers first. `names` is
        sorted before shuffling so the result does not depend on the order the
        filesystem happened to return files in.
        """
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(sorted(names)).tolist()
        n = len(shuffled)
        n_train = int(split_ratio[0] * n)
        n_val = int(split_ratio[1] * n)
        splits = {
            "train": shuffled[:n_train],
            "val":   shuffled[n_train:n_train + n_val],
            "test":  shuffled[n_train + n_val:],
        }
        if split not in splits:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
        return splits[split]
    
        



# Normalization
def compute_normalization_stats(data_dir, sample_tiles=20, subdir="ae_embeddings", out="norm_stats.json"):
    """
    Welford online mean/std over a sample of tiles.
    Works for any embedding dimensionality — infers n_channels from first file.
    """
    emb_dir = Path(data_dir) / subdir
    pattern = "*_ae.npy" if subdir == "ae_embeddings" else "*_ae.npy"

    all_paths = list(emb_dir.glob(pattern))
    if not all_paths:
        raise FileNotFoundError(f"No files matching '{pattern}' in {emb_dir}")

    # 2. Randomly sample 'sample_tiles' or the total count, whichever is smaller
    n_to_sample = min(len(all_paths), sample_tiles) if sample_tiles else len(all_paths)
    paths = random.sample(all_paths, k=n_to_sample)
    # paths = list(emb_dir.glob(pattern))[:sample_tiles]

    if not paths:
        raise FileNotFoundError(f"No files matching '{pattern}' in {emb_dir}")

    n_channels = np.load(paths[0]).shape[-1]  # infer instead of hardcoding
    count = 0 ; mean = None ; M2 = None

    for p in paths:
        tile = np.load(p).reshape(-1, n_channels).astype(np.float64)
        for row in tile:
            count += 1
            if mean is None:
                mean = row.copy()
                M2 = np.zeros(n_channels)
            else:
                delta = row - mean
                mean += delta / count
                M2 += delta * (row - mean)  # uses updated mean — correct

    std = np.sqrt(M2 / count)
    stats = {"mean": mean.tolist(), "std": std.tolist()}

    out_path = Path(data_dir) / out
    with open(out_path, "w") as f:
        json.dump(stats, f)

    print(f"Saved normalization stats ({n_channels}ch, {count} pixels) → {out_path}")
    return stats





def make_dataloaders(data_dir, patch_size=64, batch_size=32, use_ae=False, num_workers=4, split_file=None):
    """
    Returns (train_loader, val_loader, test_loader).
    Augmentation is enabled only for the training split.
    """
    train_ds = BiomassDataset(data_dir, patch_size=patch_size, split="train", split_file=split_file, use_ae=use_ae, augment=True)
    val_ds = BiomassDataset(data_dir, patch_size=patch_size, split="val",   split_file=split_file, use_ae=use_ae, augment=False)
    test_ds = BiomassDataset(data_dir, patch_size=patch_size, split="test",  split_file=split_file, use_ae=use_ae, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader






# ─────────────────────────────────────────────────────────────────────────────
# Entry point — compute stats then sanity-check the dataset
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_DIR = "data"

    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # gt = GeoTessera()
    # ee.Initialize(project="alexcloud-489214")  # already authenticated
    project = os.getenv("EE_PROJECT", "esa-cci")
    # project = os.getenv('EE_PROJECT', 'alexcloud-489214')
    ee.Initialize(project=project)

    # Step 1: compute and save normalization stats (run once)
    print("=== Computing normalization stats ===")
    # tessera_stats = compute_normalization_stats(DATA_DIR, subdir="embeddings", out="norm_stats.json")
    ae_stats = compute_normalization_stats(DATA_DIR, sample_tiles=20, subdir="ae_embeddings", out="norm_stats_ae.json")

    # Step 2: sanity check
    print("\n=== Sanity check ===")
    train_loader, val_loader, test_loader = make_dataloaders(
        DATA_DIR, patch_size=64, batch_size=8, use_ae=True, num_workers=0
    )

    x_batch, y_batch = next(iter(train_loader))
    print(f"x batch: {x_batch.shape}  dtype={x_batch.dtype}")  # (8, C, 64, 64)
    print(f"y batch: {y_batch.shape}  dtype={y_batch.dtype}")  # (8, 64, 64)
    print(f"x mean={x_batch.mean():.4f}  std={x_batch.std():.4f}")  # should be ~0, ~1
    print(f"y range=[{y_batch.min():.1f}, {y_batch.max():.1f}]")    # biomass Mg/ha





# would augmentation be useful ?












  # def __getitem__(self, idx):
    #     ti, r, c = self.index[idx]
    #     P = self.patch_size
    #     tile = self.tiles[ti]

    #     # x = tile["emb"][r:r+P, c:c+P, :].copy()   # (P, P, 128)
    #     y = tile["tgt"][r:r+P, c:c+P].copy()       # (P, P)


    #     if self.use_ae and tile["ae"] is not None:
    #         x = tile["ae"][r:r+P, c:c+P, :].copy()  
    #         if self.ae_mean is not None:
    #             x = (x - self.ae_mean) / (self.ae_std + 1e-6)
    #     else:
    #         # fallback to GeoTessera if AE missing
    #         x = tile["emb"][r:r+P, c:c+P, :].copy()
    #         # Normalize GeoTessera embedding BEFORE concatenation
    #         if self.mean is not None:
    #             x = (x - self.mean) / (self.std + 1e-6)


    #     # Augmentation (training only)
    #     if self.augment:
    #         x, y = self._augment(x, y)

    #     # HWC → CHW for PyTorch
    #     x = torch.from_numpy(x.copy()).permute(2, 0, 1).float()  # (C, P, P)
    #     y = torch.from_numpy(y.copy()).float()                    # (P, P)
    #     return x, y