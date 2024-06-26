"""
author: Min Seok Lee and Wooseok Shin
"""
import torch
import torch.nn.functional as F


def Optimizer(args, model):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


def Scheduler(args, optimizer):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_factor, patience=args.patience)
    return scheduler


def Criterion(args):
    criterion = adaptive_pixel_intensity_loss
    return criterion


def adaptive_pixel_intensity_loss(pred, mask, batch_size=1):
    w1 = torch.abs(F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) - mask)
    w2 = torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    w3 = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    

    omega = 1 + 0.5 * (w1 + w2 + w3) * mask
    bce = F.binary_cross_entropy(pred, mask, reduce=None)
    abce = (omega * bce).sum(dim=(2, 3)) / (omega + 0.5).sum(dim=(2, 3))

    inter = ((pred * mask) * omega).sum(dim=(2, 3))
    union = ((pred + mask) * omega).sum(dim=(2, 3))
    aiou = 1 - (inter + 1) / (union - inter + 1)

    mae = F.l1_loss(pred, mask, reduce=None)
    index = (omega -1).sum(dim=(2, 3))
    amae = (omega * mae).sum(dim=(2, 3)) / (omega -1).sum(dim=(2, 3))
        
    for i in range(batch_size):
        if index[i].mean == 0:
            return (0.7 * abce + 0.7 * aiou).mean()
    
    return (0.7 * abce + 0.7 * aiou + 0.7 * amae).mean()
    