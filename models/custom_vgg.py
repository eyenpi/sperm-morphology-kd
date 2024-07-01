import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, features, target_layer):
        super(VGG, self).__init__()
        self.features = features
        self.output_layers = self.get_output_layers(features)
        self.target_layer = target_layer
        self.gradients = None
        self.activation = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        result = {}
        for i in range(len(nn.ModuleList(self.features))):
            x = self.features[i](x)
            if i == self.target_layer:
                self.activation = x
                h = x.register_hook(self.activations_hook)
            if i in self.output_layers:
                result[i] = x
        return result

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.activation

    @staticmethod
    def get_output_layers(features):
        return [
            i
            for i in range(len(features) - 1)
            if isinstance(features[i + 1], nn.Conv2d)
        ] + [len(features) - 1]


def make_layers(cfg, use_bias, batch_norm):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=use_bias)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_arch(cfg, use_bias, batch_norm, target_layer):
    return VGG(make_layers(cfg, use_bias, batch_norm=batch_norm), target_layer)
