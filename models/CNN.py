import torch
import torch.nn as nn
import torchvision


class CNN(torch.nn.Module):
    def __init__(self, cnn_layers_params, transform):
        super(CNN, self).__init__()
        self.transform = transform
        self.layers = nn.ModuleList()
        for layer in cnn_layers_params:
            in_channel, out_channel, kernel, stride, padding, dilation = layer
            self.layers.append(
                nn.Conv2d(
                    in_channel, out_channel, kernel, stride, padding, dilation=dilation
                )
            )
            self.layers.append(nn.BatchNorm2d(out_channel))
            self.layers.append(nn.ReLU())

    def forward(self, x):
        if self.transform is not None:
            x = self.transform(x)
        for layer in self.layers:
            x = layer(x)
        return x
