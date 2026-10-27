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

from staged_model import SpineResUNet

# ---------------- Config ----------------
VAL_DIR = "../data/raw/03_test"
MODEL_PATH = "./models/best_model_staged.pth"
OUT_DIR = "./test_preds_staged"
WINDOW_SIZE = (128, 128, 64)
STRIDE = (64, 64, 32)
NUM_CLASSES = 29
NUM_ODD_CLASSES = 15
NUM_EVEN_CLASSES = 15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"


# ---------------- Preprocessing ----------------
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
def to_native_grid_labels(labels_1mm, native_shape):
    labels = F.interpolate(
        labels_1mm, size=tuple(native_shape), mode="nearest"
    )
    return labels.squeeze().numpy().astype(np.uint8)


# ---------------- Sliding-window inference ----------------
def _window_starts(dim, win, stride):
    if dim <= win:
        return [0]
    starts = list(range(0, dim - win + 1, stride))
    if starts[-1] != dim - win:
        starts.append(dim - win)
    return starts


@torch.no_grad()
def predict_full_volume(stage1, stage2, volume, window_size=WINDOW_SIZE, stride=STRIDE, num_classes=NUM_CLASSES):
    _, _, D0, H0, W0 = volume.shape

    pad_d = max(0, window_size[0] - D0)
    pad_h = max(0, window_size[1] - H0)
    pad_w = max(0, window_size[2] - W0)
    if pad_d or pad_h or pad_w:
        volume = F.pad(volume, (0, pad_w, 0, pad_h, 0, pad_d))
    _, _, D, H, W = volume.shape

    prob_sum = torch.zeros((1, num_classes, D, H, W), device=volume.device)
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

                vol_slice = volume[sl]

                # Stage 1 (Odd)
                logits_odd = stage1(vol_slice)
                probs_odd = F.softmax(logits_odd, dim=1)

                # Stage 2 (Even)
                stage2_input = torch.cat([vol_slice, probs_odd], dim=1)
                logits_even = stage2(stage2_input)
                probs_even = F.softmax(logits_even, dim=1)

                # Merge Probabilities into 29 classes
                merged_probs = torch.zeros((1, num_classes, window_size[0], window_size[1], window_size[2]),
                                           device=volume.device)

                # Average background predictions
                merged_probs[:, 0] = (probs_odd[:, 0] + probs_even[:, 0]) / 2.0

                # Map back to 1-28
                for i in range(1, 15):
                    merged_probs[:, i * 2 - 1] = probs_odd[:, i]  # Odd (1, 3, 5...)
                    merged_probs[:, i * 2] = probs_even[:, i]  # Even (2, 4, 6...)

                prob_sum[sl] += merged_probs
                count_map[sl] += 1

    assert count_map.min() > 0, "Uncovered voxels in sliding-window inference"
    return (prob_sum / count_map)[:, :, :D0, :H0, :W0]


# ---------------- Metrics ----------------
def compute_metrics(pred, target, num_classes=NUM_CLASSES):
    pred = pred.astype(np.int32)
    target = target.astype(np.int32)

    pred_fg = (pred > 0).astype(np.float64)
    tgt_fg = (target > 0).astype(np.float64)
    inter_fg = (pred_fg * tgt_fg).sum()
    fg_dice = (2.0 * inter_fg) / (pred_fg.sum() + tgt_fg.sum() + 1e-5)

    class_dices = []
    for c in range(1, num_classes):
        tgt_c = (target == c)
        if tgt_c.sum() == 0:
            continue

        pred_c = (pred == c)
        inter_c = (pred_c * tgt_c).sum()
        dice_c = (2.0 * inter_c) / (pred_c.sum() + tgt_c.sum() + 1e-5)
        class_dices.append(dice_c)

    vert_dice = np.mean(class_dices) if class_dices else 0.0

    return fg_dice, vert_dice


# ---------------- Visualization ----------------
def save_comparison(ct, gt, pred, slice_idx, title, path, zooms, num_classes=NUM_CLASSES):
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    fig.patch.set_facecolor("black")
    ct_s = np.rot90(ct[slice_idx])
    gt_s = np.rot90(gt[slice_idx])
    pr_s = np.rot90(pred[slice_idx])

    aspect = zooms[2] / zooms[1]

    axes[0].imshow(ct_s, cmap="gray", aspect=aspect)
    axes[0].imshow(np.ma.masked_where(gt_s == 0, gt_s), cmap="nipy_spectral", vmin=0, vmax=num_classes - 1, alpha=0.6,
                   aspect=aspect)
    axes[0].set_title(f"Ground Truth - {title} (Slice {slice_idx})", color="white")

    axes[1].imshow(ct_s, cmap="gray", aspect=aspect)
    axes[1].imshow(np.ma.masked_where(pr_s == 0, pr_s), cmap="nipy_spectral", vmin=0, vmax=num_classes - 1, alpha=0.6,
                   aspect=aspect)
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

    # Initialize both stages
    stage1_model = SpineResUNet(in_channels=1, out_channels=NUM_ODD_CLASSES).to(DEVICE)
    stage2_model = SpineResUNet(in_channels=1 + NUM_ODD_CLASSES, out_channels=NUM_EVEN_CLASSES).to(DEVICE)

    # Load shared checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    stage1_model.load_state_dict(checkpoint['stage1_state_dict'])
    stage2_model.load_state_dict(checkpoint['stage2_state_dict'])

    stage1_model.eval()
    stage2_model.eval()

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
            vol_native, vol_zooms = load_nifti(ct_path)
            gt, _ = load_nifti(mask_path)

            if gt.shape != vol_native.shape:
                raise ValueError(f"CT shape {vol_native.shape} and mask shape {gt.shape} differ on disk")

            vol_1mm = normalize_intensity(resample_volume(vol_native, vol_zooms, order=1))
            x = torch.tensor(vol_1mm, dtype=torch.float32)[None, None].to(DEVICE)

            # Pass both models into the prediction pipeline
            probs_1mm = predict_full_volume(stage1_model, stage2_model, x).cpu()

            labels_1mm = probs_1mm.argmax(dim=1, keepdim=True).float()

            pred = to_native_grid_labels(labels_1mm, gt.shape)

            fg_dice, vert_dice = compute_metrics(pred, gt)
            results.append((name, gt.shape, fg_dice, vert_dice))
            tqdm.write(f"{name}: shape={gt.shape} FG_Dice={fg_dice:.4f} Vert_Dice={vert_dice:.4f}")

            np.savez_compressed(
                os.path.join(OUT_DIR, "masks", f"{name}_pred.npz"), pred=pred
            )

            mid = gt.shape[0] // 2
            save_comparison(
                normalize_intensity(vol_native), gt, pred, mid, name,
                os.path.join(OUT_DIR, "plots", f"{name}_sagittal.png"),
                vol_zooms,
            )

            del x, probs_1mm, labels_1mm, pred
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
        writer.writerow(["volume", "shape", "fg_dice", "vert_dice"])
        for name, shape, fg, vd in results:
            writer.writerow([name, "x".join(map(str, shape)), f"{fg:.4f}", f"{vd:.4f}"])

    fg_arr = np.array([r[2] for r in results])
    vert_arr = np.array([r[3] for r in results])

    print("\n--- Two-Stage Multi-class Test Summary ---")
    print(f"Volumes processed    : {len(results)}")
    print(f"Mean Foreground Dice : {fg_arr.mean():.4f} (+/- {fg_arr.std():.4f})")
    print(f"Mean Vertebra Dice   : {vert_arr.mean():.4f} (+/- {vert_arr.std():.4f})")

    worst = min(results, key=lambda r: r[3])
    print(f"Worst volume (Naming): {worst[0]} (Vert Dice: {worst[3]:.4f})")
    print(f"Metrics saved to     : {csv_path}")


if __name__ == "__main__":
    main()