import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from multi_dataset import SpineDataset
from multi_model import SpineResUNet
import os
import matplotlib.pyplot as plt
from losses import CEDiceLoss, NUM_CLASSES, foreground_dice, hard_dice_stats
import numpy as np
from tqdm import tqdm

BATCH_SIZE = 4
LR = 0.005
EPOCHS = 100
REC_WEIGHT = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"


def make_masked_input(clean, patch=16, ratio=0.5):
    B, _, X, Y, Z = clean.shape
    keep = (torch.rand(B, 1, X // patch, Y // patch, Z // patch,
                       device=clean.device) > ratio).float()
    keep = F.interpolate(keep, size=(X, Y, Z), mode="nearest")
    return torch.cat([clean * keep, keep], dim=1), keep  # 2 channels


def save_visualization(epoch, img_tensor, mask_tensor, pred_tensor, save_dir):

    mid = (mask_tensor[0] > 0).sum(dim=(1, 2)).argmax().item()

    img_slice = np.rot90(img_tensor[0, 0, mid].cpu().numpy())
    mask_slice = np.rot90(mask_tensor[0, mid].cpu().numpy())
    pred_slice = np.rot90(pred_tensor[0].argmax(dim=0)[mid].detach().cpu().numpy())

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(img_slice, cmap="gray")
    ax[0].set_title(f"Input (Epoch {epoch}, slice {mid})")

    for a, data, title in (
        (ax[1], mask_slice, "Ground Truth"),
        (ax[2], pred_slice, "Prediction"),
    ):
        a.imshow(img_slice, cmap="gray")
        a.imshow(
            np.ma.masked_where(data == 0, data),
            cmap="nipy_spectral",
            vmin=0,
            vmax=NUM_CLASSES - 1,
            alpha=0.6,
        )
        a.set_title(title)

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch:03d}.png"))
    plt.close(fig)


def train():
    print(f"--- Starting Training on {DEVICE} ---")

    train_ds = SpineDataset(split="train", transform=True)
    val_ds = SpineDataset(split="val", transform=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    vis_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    fixed_img, fixed_mask = next(iter(vis_loader))
    fixed_img = fixed_img.to(DEVICE)
    fixed_mask = fixed_mask.to(DEVICE)

    vis_dir = "visualizations/multi_plan_ssl"
    os.makedirs(vis_dir, exist_ok=True)
    print(f"--> Visualizations will be saved to: {vis_dir}/")

    model = SpineResUNet().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=5)
    criterion = CEDiceLoss(num_classes=NUM_CLASSES, include_bg=False)

    best_val_dice = -1.0
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints/multi_ssl", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0
        epoch_train_seg = 0

        loop = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]", leave=True
        )

        for imgs, masks in loop:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)


            x, keep = make_masked_input(imgs)
            seg, rec = model(x)
            loss_seg, loss_ce, loss_dice = criterion(seg, masks)
            hidden = 1 - keep
            loss_rec = (F.l1_loss(rec, imgs, reduction="none") * hidden).sum() / (
                hidden.sum() + 1e-6
            )
            loss = loss_seg + REC_WEIGHT * loss_rec

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            epoch_train_seg += loss_seg.item()
            loop.set_postfix(
                loss=loss.item(),
                ce=loss_ce.item(),
                dice=loss_dice.item(),
                rec=loss_rec.item(),
            )

        model.eval()
        epoch_val_loss = 0
        epoch_val_fg = 0
        dice_sum = torch.zeros(NUM_CLASSES, device=DEVICE)
        dice_cnt = torch.zeros(NUM_CLASSES, device=DEVICE)

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                x, _ = make_masked_input(imgs, ratio=0)
                seg, _ = model(x, with_recon=False)

                loss, loss_ce, loss_dice = criterion(seg, masks)

                epoch_val_loss += loss.item()
                epoch_val_fg += foreground_dice(seg, masks)

                d, present = hard_dice_stats(seg, masks)
                dice_sum += d * present.float()
                dice_cnt += present.float()

            xv, _ = make_masked_input(fixed_img, ratio=0)
            vis_pred, _ = model(xv, with_recon=False)
            save_visualization(epoch + 1, fixed_img, fixed_mask, vis_pred, vis_dir)

        # per-class hard Dice over the whole val set, vertebrae only (skip background)
        per_class = dice_sum[1:] / dice_cnt[1:].clamp(min=1)
        seen = dice_cnt[1:] > 0
        avg_val_dice = per_class[seen].mean().item() if seen.any() else 0.0

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_seg = epoch_train_seg / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_fg = epoch_val_fg / len(val_loader)

        print(f"Epoch {epoch + 1} Results:")
        print(f"  Train Loss (total): {avg_train_loss:.4f}")
        print(f"  Train Loss (seg)  : {avg_train_seg:.4f}   <- compare with Val Loss")
        print(f"  Val Loss          : {avg_val_loss:.4f}")
        print(f"  Val FG Dice       : {avg_val_fg:.4f}   <- finding the spine")
        print(f"  Val Mean Dice     : {avg_val_dice:.4f}   <- vertebra labels (hard)")
        print("-" * 30)

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val_dice)
        curr_lr = optimizer.param_groups[0]["lr"]
        if curr_lr < prev_lr:
            print(f"  --> Learning Rate Reduced: {prev_lr} -> {curr_lr}")

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), "models/best_model_multi_ssl.pth")
            print("  --> New Best Model Saved!")
            print("-" * 30)

        torch.save(
            model.state_dict(), f"checkpoints/multi_ssl/model_epoch_{epoch + 1}.pth"
        )


if __name__ == "__main__":
    train()
