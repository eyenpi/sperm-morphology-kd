import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(vgg16(weights=VGG16_Weights.DEFAULT).features)
        self.output_layers = self.get_output_layers(features)
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        output = {}
        for i in range(len(self.features)):
            x = self.features[i](x)
            if i in self.output_layers:
                output[i] = x
        return output

    @staticmethod
    def get_output_layers(features):
        return [
            i
            for i in range(len(features) - 1)
            if isinstance(features[i + 1], nn.Conv2d)
        ] + [len(features) - 1]
