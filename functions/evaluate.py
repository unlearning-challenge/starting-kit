import torch
import torchvision
import requests
import functions.tools
import matplotlib.pyplot as plt
import numpy as np


@torch.no_grad()
def evaluate_model(model, device, batch_size: int = 2048, workers=6):
    # download and pre-process CIFAR10
    normalize = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root=r'F:\pycharmproject\dataset\cifar10', train=True, download=True, transform=normalize
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=workers)

    # we split held out data into test and validation set
    held_out = torchvision.datasets.CIFAR10(
        root=r'F:\pycharmproject\dataset\cifar10', train=False, download=True, transform=normalize
    )
    test_set, val_set = torch.utils.data.random_split(held_out, [0.2, 0.8])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)

    # download pre-trained weights
    response = requests.get(
        "https://unlearning-challenge.s3.eu-west-1.amazonaws.com/weights_resnet18_cifar10.pth"
    )
    open("weights_resnet18_cifar10.pth", "wb").write(response.content)

    # load model with pre-trained weights
    model.to(device)
    model.eval()

    print(f"\rTrain set accuracy: {100.0 * functions.tools.accuracy(model, train_loader, device):0.1f}%")
    print(f"\rTest set accuracy: {100.0 * functions.tools.accuracy(model, test_loader, device):0.1f}%")

    model.train()


def plot(mia_scores, mia_scores_ft, test_losses, forget_losses, ft_test_losses, ft_forget_losses):
    # From the score above, the MIA is indeed less accurate on the unlearned model than on the original model, as expected. Finally, we'll plot the histogram of losses of the unlearned model on the train and test set. From the below figure, we can observe that the distributions of forget and test losses are more similar under the unlearned model compare to the original model, as expected.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.set_title(f"Pre-trained model.\nAttack accuracy: {mia_scores.mean():0.2f}")
    ax1.hist(test_losses, density=True, alpha=0.5, bins=50, label="Test set")
    ax1.hist(forget_losses, density=True, alpha=0.5, bins=50, label="Forget set")

    ax2.set_title(f"Unlearned model.\nAttack accuracy: {mia_scores_ft.mean():0.2f}")
    ax2.hist(ft_test_losses, density=True, alpha=0.5, bins=50, label="Test set")
    ax2.hist(ft_forget_losses, density=True, alpha=0.5, bins=50, label="Forget set")

    ax1.set_xlabel("Loss")
    ax2.set_xlabel("Loss")
    ax1.set_ylabel("Frequency")
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax1.set_xlim((0, np.max(test_losses)))
    ax2.set_xlim((0, np.max(test_losses)))
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax1.legend(frameon=False, fontsize=14)
    plt.show()

def plot_origin(train_losses, test_losses):
    # plot losses on train and test set
    plt.title("Losses on train and test set (pre-trained model)")
    plt.hist(test_losses, density=True, alpha=0.5, bins=50, label="Test set")
    plt.hist(train_losses, density=True, alpha=0.5, bins=50, label="Train set")
    plt.xlabel("Loss", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xlim((0, np.max(test_losses)))
    plt.yscale("log")
    plt.legend(frameon=False, fontsize=14)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()
