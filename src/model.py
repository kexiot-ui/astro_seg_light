"""
model.py — U-Net (resnet34, imagenet) для сегментации.
Число выходных каналов (классов) задаётся параметром и берётся из чекпоинта,
поэтому одна и та же программа работает и с GFAP-моделями [mask, bound] (2 канала),
и с моделями друга серого/белого вещества [black, gray, white] (3 канала), и т.п.
N_CLASSES — значение по умолчанию (GFAP), если в чекпоинте нет ключа n_classes.
"""
import torch.nn as nn
import segmentation_models_pytorch as smp

N_CLASSES = 2  # умолчание для GFAP [mask, bound]


class AstroUNet(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet",
                 in_channels=3, n_classes=N_CLASSES):
        super().__init__()
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=n_classes,
            activation=None,
        )

    def forward(self, x):
        return self.net(x)


def build_model(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, n_classes=N_CLASSES):
    return AstroUNet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                     in_channels=in_channels, n_classes=n_classes)
