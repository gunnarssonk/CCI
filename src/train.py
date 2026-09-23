
import os, json, torch, wandb, argparse, math, random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datetime import datetime
from dataclasses import dataclass, asdict, fields
from src.dataset.dataset import BiomassDataset, compute_normalization_stats
from src.model.model import SmallCNN, PointWiseModel

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



def visualize_predictions(x, y_true, y_pred, epoch, save_dir=None):
    """
    Visualize a batch sample.
    
    Args:
        x: input tensor [B, H, W, C]
        y_true: ground truth [B, H, W] or [B, 1, H, W]
        y_pred: prediction [B, H, W] or [B, 1, H, W]
    """
    # take first sample in batch
    inp = x[0].detach().cpu()
    gt = y_true[0].detach().cpu()
    pred = y_pred[0].detach().cpu()

    # remove singleton channel if present
    if gt.ndim == 3 and gt.shape[0] == 1:
        gt = gt.squeeze(0)
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred.squeeze(0)

    # quick RGB-like visualization of input
    input_vis = inp.mean(dim=-1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(input_vis, cmap="viridis")
    ax[0].set_title("Input")

    im1 = ax[1].imshow(gt, cmap="magma")
    ax[1].set_title("Ground Truth")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)

    im2 = ax[2].imshow(pred, cmap="magma")
    ax[2].set_title("Prediction")
    plt.colorbar(im2, ax=ax[2], fraction=0.046)

    plt.suptitle(f"Epoch {epoch}")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch}.png"))

    wandb.log({
        "prediction_example": wandb.Image(fig)
    })

    plt.close(fig)

# Models selectable from the command line, by name. Config stores the *name* (a
# string) rather than the class itself: argparse can only build simple types from
# text, so a `type` field could never be set with --model_class, and asdict() had
# to serialise a class object into the wandb config. Add a new model here and it
# becomes available as --model_name straight away.
MODELS = {
    "PointWiseModel": PointWiseModel,
    "SmallCNN": SmallCNN,
}


@dataclass
class Config:
    """Configuration for the model architecture."""
    model_name: str = "PointWiseModel"   # one of MODELS above
    batch_size: int = 16
    patch_size: int = 128
    epochs: int = 2000
    lr: float = 1e-4

    # early stopping: give up after this many epochs with no new best val_loss.
    # Set to 0 to disable and always run the full `epochs`.
    patience: int = 50

    # data — logged to wandb so a run records which dataset it used
    data_dir: str = "data_gee/data_france_2020_v2"
    use_ae: bool = True
    # Standardise inputs per channel (stats from a 20-tile sample of data_dir) and
    # divide targets by target_scale so the loss is O(1) instead of O(1000). Both
    # are saved to train_meta.json next to the checkpoint; evaluate.py reads them
    # back so predictions come out in Mg/ha again. Runs before 2026-09-23 used
    # neither (normalize=False, target_scale=1).
    normalize: bool = True
    target_scale: float = 100.0
    split_train: float = 0.7
    split_val: float = 0.15
    seed: int = 42          # controls the train/val/test split; keep fixed to compare runs
    # Seed for weight init and batch order only. None → same as `seed`. Vary this
    # (and not `seed`) for repeat runs, so the test tiles stay identical and the
    # spread in the metrics is model variance, not split variance.
    init_seed: int = None

    exp_name: str = "0"
    run_name: str = "run_0"
    project_name: str = "geotessera-biomass"

    is_param: bool = False




# =============================
#       TRAINING LOOP
# =============================
def train(
    cfg: Config,
    num_workers=0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_dir = cfg.data_dir

    # Initialize wandb. The timestamp keeps run names unique even if the same
    # exp_name/run_name pair is used twice.
    stamp = datetime.now().strftime("%m%d_%H%M")
    # basename so a nested path like data_gee/data_uniform doesn't put a "/" in the name
    log_name = f"{cfg.exp_name}_{cfg.run_name}_{os.path.basename(cfg.data_dir.rstrip('/'))}_{stamp}"
    wandb.init(
        project=cfg.project_name,
        name=log_name,
        config=asdict(cfg),
        tags=[cfg.exp_name, cfg.model_name],
    )

    # =========================
    #         DATASET
    # =========================
    split_ratio = (cfg.split_train, cfg.split_val, 1.0 - cfg.split_train - cfg.split_val)

    # Fixed seed → the same tiles land in train/val/test on every run, so runs
    # are actually comparable to each other.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed if cfg.init_seed is None else cfg.init_seed)

    ae_stats = compute_normalization_stats(data_dir, sample_tiles=20, subdir="ae_embeddings", out="norm_stats_ae.json") if cfg.normalize else None

    # BiomassDataset shuffles the tile list itself and then slices it: the first 70%
    # become "train", the next 15% become "val". Because we build it twice, it would
    # shuffle twice — two different orders — and a tile sitting in the first 70% of
    # order A can easily sit in the val slice of order B. That tile would then be both
    # trained on and validated on ("leakage"), making the val score look better than
    # the model really is. Re-seeding forces both builds to use the identical order,
    # so the two slices can never overlap.

    # only initize biomass dataset once, then split into train and val using the split_ratio
    #biomass dataset would eg have a method called train_bio or val_bio that would have a version of it rather than re-create it 


    train_ds = BiomassDataset(data_dir, patch_size=cfg.patch_size, split="train", split_ratio=split_ratio, use_ae=cfg.use_ae, augment=False, seed=cfg.seed, norm_stats=ae_stats)
    random.seed(cfg.seed)
    val_ds = BiomassDataset(data_dir, patch_size=cfg.patch_size, split="val", split_ratio=split_ratio, use_ae=cfg.use_ae, augment=False, seed=cfg.seed, norm_stats=ae_stats)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print("passed dataset and loaders")

    # Record what this run actually trained on, so it shows up as columns in wandb.
    wandb.config.update({
        "n_train_tiles": len(train_ds),
        "n_val_tiles": len(val_ds),
        "n_total_tiles": len(train_ds) + len(val_ds),
    })

    # =========================
    #          MODEL
    # =========================
    sample_x, sample_y, sample_valid, sample_name = train_ds[0]
    in_channels = sample_x.shape[-1]

    if cfg.model_name not in MODELS:
        raise ValueError(f"Unknown model_name {cfg.model_name!r}. Choose one of: {', '.join(MODELS)}")
    model = MODELS[cfg.model_name](in_channels).to(device)
    wandb.watch(model, log_freq=100)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    # No nn.MSELoss() here: both loops compute the mean squared error by hand so
    # that no-data pixels can be excluded via the `valid` mask.

    print(f"Model input channels: {in_channels}")

    # Checkpoint folder is created up front, because the best model is now saved
    # during training rather than only at the end. run_dir carries the same
    # timestamp as the wandb run name, so a checkpoint can always be traced back
    # to its run (and nothing gets overwritten).
    run_dir = os.path.join("checkpoints", cfg.exp_name, f"{cfg.run_name}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "model.pth")

    # Everything evaluate.py needs to feed this checkpoint the same inputs it saw
    # in training and to turn its outputs back into Mg/ha.
    with open(os.path.join(run_dir, "train_meta.json"), "w") as f:
        json.dump({
            "model_name": cfg.model_name,
            "normalize": cfg.normalize,
            "norm_stats": ae_stats,
            "target_scale": cfg.target_scale,
            "data_dir": cfg.data_dir,
            "seed": cfg.seed,
            "init_seed": cfg.init_seed,
            "split_train": cfg.split_train,
            "split_val": cfg.split_val,
        }, f)

    # =========================
    #      TRAINING LOOP
    # =========================
    best_val_loss = float("inf")   # lowest val_loss seen so far
    best_epoch = 0
    epochs_since_best = 0          # how long since we last improved

    pbar_epoch = tqdm(range(cfg.epochs), desc="Overall Progress")
    for epoch in range(cfg.epochs):

        # ---- train ----
        model.train() ; train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for x, y, valid, _ in train_bar:
            x = x.to(device) ; y = y.to(device).squeeze(-1) / cfg.target_scale ; valid = valid.to(device)

            pred = model(x)
            # Mean squared error over usable pixels only. The dataset has already
            # zeroed the no-data pixels in x; `valid` keeps them out of the loss so
            # the model is not punished for failing to predict where there is no
            # input. Previously this used torch.isinf, which never matched, because
            # the no-data marker (-3.4e38) is finite.
            loss = (((pred - y) ** 2) * valid).sum() / valid.sum().clamp(min=1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(batch_loss=f"{loss.item():.4f}")
            wandb.log({"batch_loss": loss.item()})

        avg_train_loss =  train_loss/len(train_loader)


        # ---- validation ----
        model.eval() ; val_loss = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        with torch.no_grad():
            for i, (x, y, valid, _) in enumerate(val_bar):
                x = x.to(device) ; y = y.to(device).squeeze(-1) / cfg.target_scale ; valid = valid.to(device)

                pred = model(x)
                # same masked loss as the training loop, so the two are comparable
                loss = (((pred - y) ** 2) * valid).sum() / valid.sum().clamp(min=1)

                val_loss += loss.item()
                val_bar.set_postfix(val_loss=f"{loss.item():.4f}")

                        # store first batch for visualization
                if i == 0:
                    vis_batch = (
                        x.detach().cpu(),
                        y.detach().cpu() * cfg.target_scale,     # back to Mg/ha for the plot
                        pred.detach().cpu() * cfg.target_scale
                    )

        avg_val_loss = val_loss / len(val_loader)
        pbar_epoch.set_postfix({
            "T-Loss": f"{avg_train_loss:.4f}", 
            "V-Loss": f"{avg_val_loss:.4f}"
        })

        # ---- early stopping ----
        # Save whenever this epoch beats the best val_loss so far, so the file on
        # disk is always the best model, not just the most recent one.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_since_best = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_since_best += 1

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "best_val_loss": best_val_loss,
            "epochs_since_best": epochs_since_best,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        if vis_batch is not None and epoch % 10 == 0:
            visualize_predictions(
                vis_batch[0],
                vis_batch[1],
                vis_batch[2],
                epoch + 1,
                save_dir="visu"
            )
        
        print(f"Epoch {epoch+1}/{cfg.epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Best: {best_val_loss:.4f} (epoch {best_epoch})")

        # Val loss has not improved for `patience` epochs → more training is not
        # helping, so stop rather than burn hours overfitting.
        if cfg.patience and epochs_since_best >= cfg.patience:
            print(f"\nEarly stop at epoch {epoch+1}: no improvement for {cfg.patience} epochs. "
                  f"Best val_loss {best_val_loss:.4f} at epoch {best_epoch}.")
            break

    pbar_epoch.close()

    # The best checkpoint was already written during the loop; record how it did
    # and upload that file (not the final, possibly-overfitted weights).
    wandb.summary["best_val_loss"] = best_val_loss
    wandb.summary["best_epoch"] = best_epoch
    wandb.summary["stopped_at_epoch"] = epoch + 1

    artifact = wandb.Artifact('biomass-model', type='model')
    artifact.add_file(ckpt_path)
    wandb.log_artifact(artifact)
    wandb.finish()





def str2bool(value):
    """argparse would read the string "False" as True, so parse booleans by hand."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if value.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def args_extract(parser: argparse.ArgumentParser):
    for field in fields(Config):
        # Determine the type (handling types like 'type' carefully)
        field_type = field.type if field.type != type else None
        if field_type is bool:
            field_type = str2bool
        parser.add_argument(
            f"--{field.name}",
            type=field_type,
            default=field.default
        )

    args = parser.parse_args()
    config_keys = {f.name for f in fields(Config)}
    extra_args = set(vars(args).keys()) - config_keys
    if extra_args:
        print(f"Arguments ignored (not in Config): {', '.join(extra_args)}")

    filtered_args = {k: v for k, v in vars(args).items() if k in config_keys}
    return filtered_args








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    filtered_args = args_extract(parser)
    config = Config(**filtered_args)

    train(cfg=config)








    # fig, axes = plt.subplots(rows, 2*cols, figsize=(4*2*cols, 4*rows))
    # if rows == 1: axes = np.expand_dims(axes, axis=0)
    # if axes.ndim == 1: axes = np.expand_dims(axes, axis=0)
    # for i in range(B):
    #     row = i // cols ; col = i % cols
    #     x_sample = x_batch[i]  # (H, W, C)
    #     # mean over channels
    #     x_vis = x_sample.mean(dim=-1)
    #     # detect invalid pixels
    #     invalid_mask = torch.isinf(x_vis)

    #     # convert for plotting
    #     x_vis_np = x_vis.cpu().numpy()

    #     axes[row, 2 * col].imshow(x_vis_np, cmap='viridis')
    #     axes[row, 2 * col].axis('off')

    #     # 🚨 overlay invalid pixels in red
    #     if invalid_mask.any():
    #         mask_np = invalid_mask.cpu().numpy()
    #         axes[row, 2 * col].imshow(
    #             mask_np,
    #             cmap='Reds',
    #             alpha=0.5  # transparent overlay
    #         )
    #         axes[row, 2 * col].set_title(
    #             f"{name_batch[i]}\n⚠️ invalid pixels",
    #             color='red',
    #             fontsize=8
    #         )
    #     else:
    #         axes[row, 2 * col].set_title(name_batch[i], fontsize=8)

    #     # target
    #     y_vis = y_batch[i].squeeze(0).cpu().numpy()
    #     axes[row, 2 * col + 1].imshow(y_vis, cmap='YlGn')
    #     axes[row, 2 * col + 1].axis('off')
    
    # plt.tight_layout()
    # plt.show()




    

    # x_batch, y_batch, name_batch = next(iter(train_loader))
    # x_vis = x_batch.mean(dim=-1)
    # mask = torch.isinf(x_vis) ; mask = mask.unsqueeze(-1) ; mask = mask.expand_as(x_batch) 
    # x_batch = torch.where(mask, torch.zeros_like(x_batch), x_batch)

    # print("x batch:", x_batch.shape)  # (B, C, P, P)
    # print("y batch:", y_batch.shape)  # (B, P, P)

    # B = x_batch.shape[0] ; cols = math.ceil(math.sqrt(B)) ; rows = math.ceil(B / cols)
    
    # fig, axes = plt.subplots(rows, 2*cols, figsize=(4*2*cols, 4*rows))
    # if rows == 1: axes = np.expand_dims(axes, axis=0)
    # if axes.ndim == 1: axes = np.expand_dims(axes, axis=0)
    # for i in range(B):
    #     row = i // cols ; col = i % cols
    #     x_vis = x_batch[i].mean(dim=-1).cpu().numpy()

    #     axes[row, 2 * col].imshow(x_vis, cmap='viridis')
    #     axes[row, 2 * col].axis('off')

    #     # y_vis = y_batch[i].cpu().numpy()
    #     y_vis = y_batch[i].squeeze(0).cpu().numpy()

    #     axes[row, 2 * col + 1].imshow(y_vis, cmap='YlGn')
    #     axes[row, 2 * col + 1].axis('off')

    # plt.tight_layout()
    # plt.show()