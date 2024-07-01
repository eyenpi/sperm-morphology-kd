import torch
import torch.nn as nn


class MseDirectionLoss(nn.Module):
    def __init__(self, lamda, crit, vgg_important_layers, model_important_layers):
        super(MseDirectionLoss, self).__init__()
        self.lamda = lamda
        self.crit = crit
        self.criterion = nn.MSELoss()
        self.similarity_loss = torch.nn.CosineSimilarity()
        self.model_important_layers = model_important_layers
        self.vgg_important_layers = vgg_important_layers

    def forward(self, output_pred, output_real):
        y_pred = [output_pred[layer] for layer in self.model_important_layers[-4:]]
        y_real = [output_real[layer] for layer in self.vgg_important_layers[-4:]]

        abs_losses = [self.criterion(pred, real) for pred, real in zip(y_pred, y_real)]
        dir_losses = [
            torch.mean(
                1
                - self.similarity_loss(
                    pred.view(pred.shape[0], -1), real.view(real.shape[0], -1)
                )
            )
            for pred, real in zip(y_pred, y_real)
        ]

        total_loss = sum(dir_losses[-self.crit :]) + self.lamda * sum(
            abs_losses[-self.crit :]
        )

        return total_loss
