import torch
import numpy as np


def compute_losses(net, loader, device):
    """Auxiliary function to compute per-sample losses"""

    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        logits = net(inputs)
        losses = criterion(logits, targets).numpy(force=True)
        for l in losses:
            all_losses.append(l)

    return np.array(all_losses)