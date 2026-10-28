import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # Added for logging
from staged_dataset import SpineDataset
from staged_model import SpineResUNet
import os
import matplotlib.pyplot as plt
from losses import CEDiceLoss, NUM_CLASSES
import numpy as np
from tqdm import tqdm

BATCH_SIZE = 4
LR = 0.001
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"

NUM_ODD_CLASSES = 15  # Background + 14 odd vertebrae
NUM_EVEN_CLASSES = 15  # Background + 14 even vertebrae


def save_visualization(epoch, img_tensor, mask_tensor, pred_tensor, save_dir):
    mid = mask_tensor[0].sum(dim=(1, 2)).argmax().item()

    img_slice = np.rot90(img_tensor[0, 0, mid].cpu().numpy())
    mask_slice = np.rot90(mask_tensor[0, mid].cpu().numpy())

    # pred_tensor is passed in already argmaxed and merged in this updated script
    pred_slice = np.rot90(pred_tensor[0, mid].detach().cpu().numpy())

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


def merge_predictions(odd_logits, even_logits):
    """Maps 0-14 odd/even predictions back to the unified 0-28 class space."""
    odd_pred = odd_logits.argmax(dim=1)
    even_pred = even_logits.argmax(dim=1)

    # Fixed syntax error here
    odd_mapped = torch.where(odd_pred > 0, odd_pred * 2 - 1, torch.tensor(0, device=DEVICE))
    even_mapped = torch.where(even_pred > 0, even_pred * 2, torch.tensor(0, device=DEVICE))

    return torch.max(odd_mapped, even_mapped)


def train():
    print(f"--- Starting Two-Stage Training on {DEVICE} ---")

    # Initialize TensorBoard Writer
    log_dir = "../logs/staged_plan"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"--> TensorBoard logs will be saved to: {log_dir}/")

    train_ds = SpineDataset(split="train", transform=True)
    val_ds = SpineDataset(split="val", transform=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    vis_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    fixed_img, fixed_odd_mask, fixed_even_mask = next(iter(vis_loader))
    fixed_img = fixed_img.to(DEVICE)

    # Fixed syntax error here
    fixed_mask = torch.max(
        torch.where(fixed_odd_mask > 0, fixed_odd_mask * 2 - 1, 0),
        torch.where(fixed_even_mask > 0, fixed_even_mask * 2, 0)
    ).to(DEVICE)

    vis_dir = "../visualizations/staged_plan"
    os.makedirs(vis_dir, exist_ok=True)
    print(f"--> Visualizations will be saved to: {vis_dir}/")

    # --- Initialize Both Stages ---
    stage1_model = SpineResUNet(in_channels=1, out_channels=NUM_ODD_CLASSES).to(DEVICE)
    stage2_model = SpineResUNet(in_channels=1 + NUM_ODD_CLASSES, out_channels=NUM_EVEN_CLASSES).to(DEVICE)

    # binary_model_path = "../models/best_model_ssl.pth"
    # if os.path.exists(binary_model_path):
    #     print(f"Loading binary weights into both stages from {binary_model_path}...")
    #     state = torch.load(binary_model_path, map_location=DEVICE)
    #     state = {k: v for k, v in state.items() if not k.startswith("final.")}
    #
    #     state_stage2 = {k: v for k, v in state.items() if not k.startswith("enc1.conv1.")}
    #
    #     stage1_model.load_state_dict(state, strict=False)
    #     stage2_model.load_state_dict(state_stage2, strict=False)
    # else:
    #     print(f"WARNING: Binary checkpoint not found at {binary_model_path}. Training from scratch.")

    optimizer = optim.Adam(list(stage1_model.parameters()) + list(stage2_model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)

    criterion_odd = CEDiceLoss(num_classes=NUM_ODD_CLASSES, include_bg=False)
    criterion_even = CEDiceLoss(num_classes=NUM_EVEN_CLASSES, include_bg=False)

    best_val_loss = float("inf")
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../checkpoints/staged_plan", exist_ok=True)

    global_step = 0

    for epoch in range(EPOCHS):
        stage1_model.train()
        stage2_model.train()
        epoch_train_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]", leave=True)

        for imgs, odd_masks, even_masks in loop:
            imgs = imgs.to(DEVICE)
            odd_masks = odd_masks.to(DEVICE)
            even_masks = even_masks.to(DEVICE)

            # --- STAGE 1 ---
            pred_odd_logits = stage1_model(imgs)
            loss1, _, _ = criterion_odd(pred_odd_logits, odd_masks)

            # --- MASK TRANSFER ---
            pred_odd_probs = torch.softmax(pred_odd_logits, dim=1)
            stage2_input = torch.cat([imgs, pred_odd_probs.detach()], dim=1)

            # --- STAGE 2 ---
            pred_even_logits = stage2_model(stage2_input)
            loss2, _, _ = criterion_even(pred_even_logits, even_masks)

            total_loss = loss1 + loss2

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_train_loss += total_loss.item()

            # Log running batch loss
            writer.add_scalar("Loss/Running_Train", total_loss.item(), global_step)
            global_step += 1

            loop.set_postfix(loss=total_loss.item())

        stage1_model.eval()
        stage2_model.eval()
        epoch_val_loss = 0
        epoch_val_fg = 0

        with torch.no_grad():
            for imgs, odd_masks, even_masks in val_loader:
                imgs = imgs.to(DEVICE)
                odd_masks = odd_masks.to(DEVICE)
                even_masks = even_masks.to(DEVICE)

                pred_odd_logits = stage1_model(imgs)
                loss1, _, _ = criterion_odd(pred_odd_logits, odd_masks)

                pred_odd_probs = torch.softmax(pred_odd_logits, dim=1)
                stage2_input = torch.cat([imgs, pred_odd_probs], dim=1)

                pred_even_logits = stage2_model(stage2_input)
                loss2, _, _ = criterion_even(pred_even_logits, even_masks)

                epoch_val_loss += (loss1.item() + loss2.item())

                fg_pred = (pred_odd_logits.argmax(dim=1) > 0) | (pred_even_logits.argmax(dim=1) > 0)
                fg_tgt = (odd_masks > 0) | (even_masks > 0)
                inter = (fg_pred * fg_tgt).sum().float()

                # Fixed syntax error here
                epoch_val_fg += ((2.0 * inter + 1.0) / (fg_pred.sum() + fg_tgt.sum() + 1.0)).item()

            fix_odd_logits = stage1_model(fixed_img)
            fix_odd_probs = torch.softmax(fix_odd_logits, dim=1)
            fix_stage2_input = torch.cat([fixed_img, fix_odd_probs], dim=1)
            fix_even_logits = stage2_model(fix_stage2_input)

            merged_pred = merge_predictions(fix_odd_logits, fix_even_logits)
            save_visualization(epoch + 1, fixed_img, fixed_mask, merged_pred, vis_dir)

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_fg = epoch_val_fg / len(val_loader)
        curr_lr = optimizer.param_groups[0]["lr"]

        # Log epoch-level metrics
        writer.add_scalar("Loss/Epoch_Train", avg_train_loss, epoch + 1)
        writer.add_scalar("Loss/Epoch_Val", avg_val_loss, epoch + 1)
        writer.add_scalar("Metric/Val_FG_Dice", avg_val_fg, epoch + 1)
        writer.add_scalar("Hyperparameters/Learning_Rate", curr_lr, epoch + 1)

        print(f"Epoch {epoch + 1} Results:")
        print(f"  Train Loss : {avg_train_loss:.4f}")
        print(f"  Val Loss   : {avg_val_loss:.4f}")
        print(f"  Val FG Dice: {avg_val_fg:.4f}   <- finding the spine")

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val_loss)
        if optimizer.param_groups[0]["lr"] < prev_lr:
            print(f"  --> Learning Rate Reduced: {prev_lr} -> {optimizer.param_groups[0]['lr']}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'stage1_state_dict': stage1_model.state_dict(),
                'stage2_state_dict': stage2_model.state_dict()
            }, "../models/best_model_staged.pth")
            print("  --> New Best Model Saved!")

        torch.save({
            'stage1_state_dict': stage1_model.state_dict(),
            'stage2_state_dict': stage2_model.state_dict()
        }, f"../checkpoints/staged_plan/model_epoch_{epoch + 1}.pth")

    writer.close()
    print("Training complete. Run `tensorboard --logdir=../logs/staged_plan` to view logs.")


if __name__ == "__main__":
    train()