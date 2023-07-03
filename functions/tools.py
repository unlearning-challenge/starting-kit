import processingtools as pt
import torch
import numpy as np
from sklearn import linear_model, model_selection
import torchvision
from torchvision import transforms
import torch.utils.data
import torchvision.models


def accuracy(net, loader, device):
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0
    for inputs, targets in pt.ProgressBar(loader, total=len(loader), finish_mark=None):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('\r', end='')
    return correct / total


def unlearning(net, retain, forget, validation, device, recoder, epochs: int = 5):
    """Unlearning by fine-tuning.

    Fine-tuning is a very simple algorithm that trains using only
    the retain set.

    Args:
      net : nn.Module.
        pre-trained model to use as base of unlearning.
      retain : torch.utils.data.DataLoader.
        Dataset loader for access to the retain set. This is the subset
        of the training set that we don't want to forget.
      forget : torch.utils.data.DataLoader.
        Dataset loader for access to the forget set. This is the subset
        of the training set that we want to forget. This method doesn't
        make use of the forget set.
      validation : torch.utils.data.DataLoader.
        Dataset loader for access to the validation set. This method doesn't
        make use of the validation set.
      device: device info
      epochs: epochs
    Returns:
      net : updated model
    """

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    net.train()

    loss = 0
    count = 0
    losses_sum = 0
    for _ in range(epochs):
        for inputs, targets in pt.ProgressBar(retain, total=len(retain), finish_mark=None, detail_func=lambda _: f'{float(loss):.05f}'):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            losses_sum = losses_sum + float(loss)
            count = count + 1

        print('\r', end='')
        recoder.print(f'[{_ + 1}/{epochs}] loss: {float(losses_sum / count):0.5f}')
        scheduler.step()

    net.eval()
    return net


def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    """Computes cross-validation score of a membership inference attack.

    Args:
      sample_loss : array_like of shape (n,).
        objective function evaluated on n samples.
      members : array_like of shape (n,),
        whether a sample was used for training.
      n_splits: int
        number of splits to use in the cross-validation.
    Returns:
      scores : array_like of size (n_splits,)
    """

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )


def get_loader(workers: int = 6, train_batch: int = 2048, test_batch: int = 4096):
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root=r'F:\pycharmproject\dataset\cifar10', train=True, download=True, transform=normalize
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch, shuffle=False, num_workers=workers)

    # we split held out data into test and validation set
    held_out = torchvision.datasets.CIFAR10(
        root=r'F:\pycharmproject\dataset\cifar10', train=False, download=True, transform=normalize
    )
    test_set, val_set = torch.utils.data.random_split(held_out, [0.2, 0.8])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch, shuffle=False, num_workers=workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=test_batch, shuffle=False, num_workers=workers)

    forget_set, retain_set = torch.utils.data.random_split(train_set, [0.1, 0.9])
    forget_loader = torch.utils.data.DataLoader(
        forget_set, batch_size=train_batch, shuffle=False, num_workers=workers
    )
    retain_loader = torch.utils.data.DataLoader(
        retain_set, batch_size=train_batch, shuffle=False, num_workers=workers
    )
    print()

    return train_loader, test_loader, val_loader, retain_loader, forget_loader


def get_model(device):
    model = torchvision.models.resnet18(weights=None, num_classes=10)
    model_ft = torchvision.models.resnet18(weights=None, num_classes=10)

    model.load_state_dict(torch.load("weights_resnet18_cifar10.pth", map_location=device))
    model_ft.load_state_dict(torch.load("weights_resnet18_cifar10.pth", map_location=device))

    model.to(device)
    model_ft.to(device)

    return model, model_ft
