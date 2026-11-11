import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 29

# ---------------- Dice ----------------
def per_class_dice(logits, target, num_classes=NUM_CLASSES, eps=1.0, soft=True):

    if soft:
        probs = F.softmax(logits, dim=1)
    else:
        pred = logits.argmax(dim=1)

    dice = torch.zeros(num_classes, device=logits.device)
    present = torch.zeros(num_classes, dtype=torch.bool, device=logits.device)

    for c in range(num_classes):
        tgt_c = (target == c).float()
        tgt_sum = tgt_c.sum()
        if tgt_sum == 0:
            continue
        present[c] = True

        p_c = probs[:, c] if soft else (pred == c).float()
        inter = (p_c * tgt_c).sum()
        dice[c] = (2 * inter + eps) / (p_c.sum() + tgt_sum + eps)

    return dice, present


def dice_loss(logits, target, num_classes=NUM_CLASSES, eps=1.0, include_bg=False):

    dice, present = per_class_dice(logits, target, num_classes, eps, soft=True)

    if not include_bg:
        present[0] = False

    if not present.any():
        return logits.sum() * 0.0

    return 1 - dice[present].mean()


# ---------------- Combined loss ----------------
class CEDiceLoss(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES, include_bg=False, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.include_bg = include_bg
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, target):
        loss_ce = self.ce(logits, target)
        loss_dice = dice_loss(logits, target, self.num_classes,
                              include_bg=self.include_bg)
        return loss_ce + loss_dice, loss_ce.detach(), loss_dice.detach()


# ---------------- Reporting metrics ----------------
@torch.no_grad()
def hard_dice_stats(logits, target, num_classes=NUM_CLASSES):

    return per_class_dice(logits, target, num_classes, soft=False)


@torch.no_grad()
def foreground_dice(logits, target, num_classes=NUM_CLASSES):

    pred_fg = (logits.argmax(dim=1) > 0).float()
    tgt_fg = (target > 0).float()
    inter = (pred_fg * tgt_fg).sum()
    return ((2 * inter + 1.0) / (pred_fg.sum() + tgt_fg.sum() + 1.0)).item()