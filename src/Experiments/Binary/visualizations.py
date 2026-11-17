import os
import glob
import csv
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import scipy.ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import SpineResUNet

# ---------------- Config ----------------
VAL_DIR = "../data/raw/03_test"
MODEL_PATH = "../../models/best_model.pth"
OUT_DIR = "../../test_preds"
WINDOW_SIZE = (128, 128, 64)   # same as PATCH_SIZE in preprocessing
STRIDE = (64, 64, 32)
THRESHOLD = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"


# ---------------- Preprocessing (identical to training) ----------------
def load_nifti(path):
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    return img.get_fdata(), img.header.get_zooms()


def resample_volume(data, current_zooms, target_zooms=(1.0, 1.0, 1.0), order=1):
    resize_factor = np.array(current_zooms) / np.array(target_zooms)
    new_shape = np.round(data.shape * resize_factor)
    real_resize_factor = new_shape / data.shape
    return scipy.ndimage.zoom(data, real_resize_factor, order=order)


def normalize_intensity(vol):
    min_hu, max_hu = -1000, 2000
    vol = np.clip(vol, min_hu, max_hu)
    return (vol - min_hu) / (max_hu - min_hu)


# ---------------- Back to the original grid ----------------
def to_native_grid(probs_1mm, native_shape):
    """
    Resample a 1 mm probability map back to the original scan's voxel grid.
    probs_1mm: (1, 1, D, H, W) tensor on CPU.
    align_corners=True matches scipy.ndimage.zoom's default (grid_mode=False),
    so this undoes the forward resampling without a half-voxel shift.
    """
    probs = F.interpolate(
        probs_1mm, size=tuple(native_shape), mode="trilinear", align_corners=True
    )
    return probs.squeeze().numpy()


# ---------------- Sliding-window inference ----------------
def _window_starts(dim, win, stride):
    """Start indices covering [0, dim) fully; last window anchored at dim - win."""
    if dim <= win:
        return [0]
    starts = list(range(0, dim - win + 1, stride))
    if starts[-1] != dim - win:
        starts.append(dim - win)
    return starts


@torch.no_grad()
def predict_full_volume(model, volume, window_size=WINDOW_SIZE, stride=STRIDE):
    """volume: (1, 1, D, H, W) tensor. Returns probability tensor of the same shape."""
    _, _, D0, H0, W0 = volume.shape

    pad_d = max(0, window_size[0] - D0)
    pad_h = max(0, window_size[1] - H0)
    pad_w = max(0, window_size[2] - W0)
    if pad_d or pad_h or pad_w:
        volume = F.pad(volume, (0, pad_w, 0, pad_h, 0, pad_d))
    _, _, D, H, W = volume.shape

    prob_sum = torch.zeros((1, 1, D, H, W), device=volume.device)
    count_map = torch.zeros((1, 1, D, H, W), device=volume.device)

    for d in _window_starts(D, window_size[0], stride[0]):
        for h in _window_starts(H, window_size[1], stride[1]):
            for w in _window_starts(W, window_size[2], stride[2]):
                sl = (
                    slice(None), slice(None),
                    slice(d, d + window_size[0]),
                    slice(h, h + window_size[1]),
                    slice(w, w + window_size[2]),
                )
                prob_sum[sl] += model(volume[sl])
                count_map[sl] += 1

    assert count_map.min() > 0, "Uncovered voxels in sliding-window inference"
    return (prob_sum / count_map)[:, :, :D0, :H0, :W0]


# ---------------- Metrics ----------------
def dice_score(pred, target, smooth=1.0):
    pred = pred.reshape(-1).astype(np.float64)
    target = target.reshape(-1).astype(np.float64)
    inter = (pred * target).sum()
    return (2.0 * inter + smooth) / (pred.sum() + target.sum() + smooth)


# ---------------- Visualization ----------------
def save_comparison(ct, gt, pred, slice_idx, title, path, zooms):
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    fig.patch.set_facecolor("black")
    ct_s = np.rot90(ct[slice_idx])
    gt_s = np.rot90(gt[slice_idx])
    pr_s = np.rot90(pred[slice_idx])

    # Native voxels are often not cubes, so stretch the display to real proportions:
    # after rot90, rows run head-to-foot (axis 2) and columns front-to-back (axis 1).
    aspect = zooms[2] / zooms[1]

    axes[0].imshow(ct_s, cmap="gray", aspect=aspect)
    axes[0].imshow(np.ma.masked_where(gt_s == 0, gt_s), cmap="autumn", alpha=0.5, aspect=aspect)
    axes[0].set_title(f"Ground Truth - {title} (Slice {slice_idx})", color="white")
    axes[1].imshow(ct_s, cmap="gray", aspect=aspect)
    axes[1].imshow(np.ma.masked_where(pr_s == 0, pr_s), cmap="winter", alpha=0.5, aspect=aspect)
    axes[1].set_title(f"Prediction - {title} (Slice {slice_idx})", color="white")
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# ---------------- Main ----------------
def main():
    print(f"--- Test inference on {DEVICE} ---")
    os.makedirs(os.path.join(OUT_DIR, "masks"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "plots"), exist_ok=True)

    model = SpineResUNet(in_channels=1, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_files = sorted(glob.glob(os.path.join(VAL_DIR, "**/*.nii.gz"), recursive=True))
    ct_files = [f for f in all_files if not f.endswith("_seg.nii.gz")]
    print(f"Found {len(ct_files)} test CT volumes in {VAL_DIR}")

    results = []
    for ct_path in tqdm(ct_files, desc="Volumes"):
        name = os.path.basename(ct_path).split(".")[0]
        mask_path = ct_path.replace(".nii.gz", "_seg.nii.gz")
        if not os.path.exists(mask_path):
            print(f"Skipping {name}: mask not found.")
            continue

        try:
            # 1. Load the scan and the ground truth exactly as they are on disk.
            vol_native, vol_zooms = load_nifti(ct_path)
            gt, _ = load_nifti(mask_path)

            # The only change to the ground truth: vertebra labels -> spine (1) / background (0).
            # No resampling, no cropping.
            gt = (gt > 0).astype(np.float32)

            if gt.shape != vol_native.shape:
                raise ValueError(
                    f"CT shape {vol_native.shape} and mask shape {gt.shape} differ on disk"
                )

            # 2. Give the model what it was trained on: a 1 mm, normalised volume.
            vol_1mm = normalize_intensity(resample_volume(vol_native, vol_zooms, order=1))
            x = torch.tensor(vol_1mm, dtype=torch.float32)[None, None].to(DEVICE)
            probs_1mm = predict_full_volume(model, x).cpu()

            # 3. Move the prediction (not the ground truth) back to the original grid.
            probs = to_native_grid(probs_1mm, gt.shape)
            pred = (probs > THRESHOLD).astype(np.uint8)

            # 4. Score against the untouched ground truth.
            soft_dice = dice_score(probs, gt)
            hard_dice = dice_score(pred, gt)
            results.append((name, gt.shape, soft_dice, hard_dice))
            tqdm.write(f"{name}: shape={gt.shape} soft_dice={soft_dice:.4f} dice={hard_dice:.4f}")

            np.savez_compressed(
                os.path.join(OUT_DIR, "masks", f"{name}_pred.npz"), pred=pred
            )

            # Middle sagittal slice, shown on the original scan
            mid = gt.shape[0] // 2
            save_comparison(
                normalize_intensity(vol_native), gt, pred, mid, name,
                os.path.join(OUT_DIR, "plots", f"{name}_sagittal.png"),
                vol_zooms,
            )

            del x, probs_1mm, probs
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing {name}: {e}")

    if not results:
        print("No volumes processed.")
        return

    csv_path = os.path.join(OUT_DIR, "test_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["volume", "shape", "soft_dice", "dice"])
        for name, shape, sd, hd in results:
            writer.writerow([name, "x".join(map(str, shape)), f"{sd:.4f}", f"{hd:.4f}"])

    hard = np.array([r[3] for r in results])
    print("\n--- Test Summary (scored on original ground-truth masks) ---")
    print(f"Volumes processed : {len(results)}")
    print(f"Mean Dice         : {hard.mean():.4f} (+/- {hard.std():.4f})")
    print(f"Worst / Best      : {hard.min():.4f} / {hard.max():.4f}")
    worst = min(results, key=lambda r: r[3])
    print(f"Worst volume      : {worst[0]}")
    print(f"Metrics saved to  : {csv_path}")


if __name__ == "__main__":
    main()