import requests
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.models import resnet18
import functions.tools
import functions.losses


if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on device:", DEVICE.upper())

    workers = 6
    batch_size = 2048

    # download and pre-process CIFAR10
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root=r'F:\pycharmproject\dataset\cifar10', train=True, download=True, transform=normalize
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=workers)

    # we split held out data into test and validation set
    held_out = torchvision.datasets.CIFAR10(
        root=r'F:\pycharmproject\dataset\cifar10', train=False, download=True, transform=normalize
    )
    test_set, val_set = torch.utils.data.random_split(held_out, [0.2, 0.8])
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)

    # download pre-trained weights
    response = requests.get(
        "https://unlearning-challenge.s3.eu-west-1.amazonaws.com/weights_resnet18_cifar10.pth"
    )
    open("weights_resnet18_cifar10.pth", "wb").write(response.content)
    weights_pretrained = torch.load("weights_resnet18_cifar10.pth", map_location=DEVICE)

    # load model with pre-trained weights
    model = resnet18(weights=None, num_classes=10)
    model.load_state_dict(weights_pretrained)
    model.to(DEVICE)
    model.eval()

    print(f"\rTrain set accuracy: {100.0 * functions.tools.accuracy(model, train_loader, DEVICE):0.1f}%")
    print(f"\rTest set accuracy: {100.0 * functions.tools.accuracy(model, test_loader, DEVICE):0.1f}%")

    forget_set, retain_set = torch.utils.data.random_split(train_set, [0.1, 0.9])
    forget_loader = torch.utils.data.DataLoader(
        forget_set, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    retain_loader = torch.utils.data.DataLoader(
        retain_set, batch_size=batch_size, shuffle=False, num_workers=workers
    )

    # The goal of an unlearning algorithm is to produce a model that approximates as much as possible the model trained solely on the retain set.
    #
    # Below is a simple unlearning algorithms provided for illustration purposes. We call this algorithm _unlearning by fine-tuning_. It starts from the pre-trained and optimizes for a few epochs on the retain set. This is a very simple unlearning algorithm, but it is not very computationally efficient.
    #
    # To make a new entry in the competitions, participants will submit an unlearning function with the same API as the one below. Note that the unlearning function takes as input a pre-trained model, a retain set, a forget set and an evaluation set (even though the fine-tuning algorithm below only uses the retain set and ignores the other datasets).

    # In[45]:

    model_ft = resnet18(weights=None, num_classes=10)
    model_ft.load_state_dict(weights_pretrained)
    model_ft.to(DEVICE)

    # Execute the unlearing routine. This might take a few minutes.
    # If run on colab, be sure to be running it on  an instance with GPUs
    model_ft = functions.tools.unlearning(model_ft, retain_loader, forget_loader, test_loader, DEVICE)


    # We have now an unlearned model `model_ft`. Besides the forgetting quality (which we'll discuss in the next section), a good unlearning algorithm should retain as much as possible the accuracy on the retain and test set.
    #
    # To quantify this potential loss of utility, we'll now compute the retain and test accuracies using the unlearned model

    # In[49]:

    print(f"\rRetain set accuracy: {100.0 * functions.tools.accuracy(model_ft, retain_loader, DEVICE):0.1f}%")
    print(f"\rTest set accuracy: {100.0 * functions.tools.accuracy(model_ft, test_loader, DEVICE):0.1f}%")


    # # 🏅 Evaluation
    #
    # In this section we'll quantify the quality of the unlearning algorithm through a simple membership inference attack (MIA). We provide this simple MIA for convenience, so that participants can quickly obtain a metric for their unlearning algorithm,but submissions will be scored using a different method.
    #
    # This MIA consists of a [logistic regression model](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) that predicts whether the model was trained on a particular sample from that sample's loss. To get an idea on the difficulty of this problem, we first plot below a histogram of the losses of the pre-trained model on the train and test set

    train_losses = functions.losses.compute_losses(model, train_loader, DEVICE)
    test_losses = functions.losses.compute_losses(model, test_loader, DEVICE)

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

    # As per the above plot, the distributions of losses are quite different between the train and test sets, as expected. In what follows, we will define an MIA that leverages the fact that examples that were trained on have smaller losses compared to examples that weren't. Using this fact, the simple MIA defined below will aim to infer whether the forget set was in fact part of the training set.
    #
    # This MIA is defined below. It takes as input the per-sample losses of the unlearned model on forget and test examples, and a membership label (0 or 1) indicating which of those two groups each sample comes from. It then returns the cross-validation accuracy of a linear model trained to distinguish between the two classes.
    #
    # Intuitively, an unlearning algorithm is successful with respect to this simple metric if the attacker isn't able to distinguish the forget set from the test set (associated with an attacker accuracy around random chance).

    forget_losses = functions.losses.compute_losses(model, forget_loader, DEVICE)

    # Since we have more forget losses than test losses, sub-sample them, to have a class-balanced dataset.
    np.random.shuffle(forget_losses)
    forget_losses = forget_losses[:len(test_losses)]

    samples_mia = np.concatenate((test_losses, forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(test_losses) + [1] * len(forget_losses)

    mia_scores = functions.tools.simple_mia(samples_mia, labels_mia)

    print(
        f"The MIA attack has an accuracy of {mia_scores.mean():.3f} on forgotten vs unseen images"
    )

    # We'll now compute the accuracy of the MIA on the unlearned model. We expect the MIA to be less accurate on the unlearned model than on the original model, since the original model has not undergone a procedure to unlearn the forget set.
    ft_forget_losses = functions.losses.compute_losses(model_ft, forget_loader, DEVICE)
    ft_test_losses = functions.losses.compute_losses(model_ft, test_loader, DEVICE)

    # Since we have more forget losses than test losses, sub-sample them, to have a class-balanced dataset.
    np.random.shuffle(ft_forget_losses)
    ft_forget_losses = ft_forget_losses[:len(ft_test_losses)]

    samples_mia_ft = np.concatenate((ft_test_losses, ft_forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(ft_test_losses) + [1] * len(ft_forget_losses)

    mia_scores_ft = functions.tools.simple_mia(samples_mia_ft, labels_mia)

    print(
        f"The MIA attack has an accuracy of {mia_scores_ft.mean():.3f} on forgotten vs unseen images"
    )

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

