import torch
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import roc_curve, auc
from models.vgg import Vgg16
from models.custom_vgg import make_arch
from utils.data_loader import load_data
from config import *


def detection_test(
    model,
    vgg,
    test_dataloader,
    vgg_important_layers,
    model_important_layers,
    crit,
    lamda,
):
    target_class = NORMAL_CLASS
    similarity_loss = torch.nn.CosineSimilarity()
    label_score = []
    model.eval()

    with torch.no_grad():
        for data in test_dataloader:
            X, Y = data
            if X.shape[1] == 1:
                X = X.repeat(1, 3, 1, 1)
            X = X.to(DEVICE)
            output_pred = model.forward(X)
            output_real = vgg(X)

            losses = []
            for i in range(4):
                y_pred = output_pred[model_important_layers[-4 + i]]
                y_real = output_real[vgg_important_layers[-4 + i]]
                abs_loss = torch.mean((y_pred - y_real) ** 2, dim=(1, 2, 3))
                dir_loss = 1 - similarity_loss(
                    y_pred.view(y_pred.shape[0], -1), y_real.view(y_real.shape[0], -1)
                )
                losses.append((abs_loss, dir_loss))

            total_loss = sum([l[1] for l in losses[-crit:]]) + lamda * sum(
                [l[0] for l in losses[-crit:]]
            )
            label_score += list(
                zip(Y.cpu().numpy().tolist(), total_loss.cpu().numpy().tolist())
            )

    labels, scores = zip(*label_score)
    labels = np.array(labels)
    labels = np.where(labels == target_class, 1, 0)
    scores = np.array(scores)
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=0)
    roc_auc = auc(fpr, tpr)
    return round(roc_auc, 4)


def test():
    _, _, test_dataloader = load_data(
        batch_size=BATCH_SIZE,
        just_normal=JUST_NORMAL,
        normal_class=NORMAL_CLASS,
        augmentation=AUGMENTATION,
        mode=MODE,
    )

    vgg = Vgg16().to(DEVICE)
    model = make_arch(
        CFG, use_bias=USE_BIAS, batch_norm=BATCH_NORM, target_layer=TARGET_LAYER
    ).to(DEVICE)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    roc_auc = detection_test(
        model,
        vgg,
        test_dataloader,
        VGG_IMPORTANT_LAYERS,
        MODEL_IMPORTANT_LAYERS,
        CRIT,
        LAMBDA,
    )
    print(f"Test ROC AUC: {roc_auc}")


if __name__ == "__main__":
    test()
