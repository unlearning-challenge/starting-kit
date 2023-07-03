import numpy as np
import torch
import functions.tools
import functions.losses
import functions.evaluate
import argparse
import processingtools as pt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../dataset/cifar10')
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='../outputs/test')
    args = pt.EnvReco.arg2abs(parser.parse_args())

    recoder = pt.EnvReco(args.save_path)
    recoder.record_arg(recoder.arg2abs(args))
    recoder.record_code()
    recoder.record_gpu()
    recoder.put_space()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # get data loaders
    _, test_loader, _, retain_loader, forget_loader = functions.tools.get_loader(args.workers, args.batch_size, args.batch_size * 4)

    # get models
    model, model_ft = functions.tools.get_model(DEVICE)

    # Execute the unlearing routine. This might take a few minutes.
    # If run on colab, be sure to be running it on  an instance with GPUs
    model_ft = functions.tools.unlearning(model_ft, retain_loader, forget_loader, test_loader, DEVICE, recoder, args.epoch)


    # We have now an unlearned model `model_ft`. Besides the forgetting quality (which we'll discuss in the next section), a good unlearning algorithm should retain as much as possible the accuracy on the retain and test set.
    #
    # To quantify this potential loss of utility, we'll now compute the retain and test accuracies using the unlearned model

    recoder.put_space()
    recoder.print(f"Retain set accuracy: {100.0 * functions.tools.accuracy(model_ft, retain_loader, DEVICE):0.1f}%")
    recoder.print(f"Test set accuracy: {100.0 * functions.tools.accuracy(model_ft, test_loader, DEVICE):0.1f}%")

    # # 🏅 Evaluation
    #
    # In this section we'll quantify the quality of the unlearning algorithm through a simple membership inference attack (MIA). We provide this simple MIA for convenience, so that participants can quickly obtain a metric for their unlearning algorithm,but submissions will be scored using a different method.
    #
    # This MIA consists of a [logistic regression model](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) that predicts whether the model was trained on a particular sample from that sample's loss. To get an idea on the difficulty of this problem, we first plot below a histogram of the losses of the pre-trained model on the train and test set

    test_losses = functions.losses.compute_losses(model, test_loader, DEVICE)

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

    recoder.put_space()
    recoder.print(
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

    recoder.print(
        f"The MIA attack has an accuracy of {mia_scores_ft.mean():.3f} on forgotten vs unseen images"
    )

    functions.evaluate.plot(mia_scores, mia_scores_ft, test_losses, forget_losses, ft_test_losses, ft_forget_losses)
