from dataclasses import dataclass
import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import ResNet as TorchResNet


def load_resnet_model(model_name: str, weights_name: str) -> TorchResNet:
    match model_name:
        case "resnext50":
            return torchvision.models.resnext50_32x4d(weights=weights_name)
        case "resnet50":
            return torchvision.models.resnet50(weights=weights_name)
        case "resnet101":
            return torchvision.models.resnet101(weights=weights_name)
        case "resnet152":
            return torchvision.models.resnet152(weights=weights_name)
        case "resnet34":
            return torchvision.models.resnet34(weights=weights_name)
        case "resnet18":
            return torchvision.models.resnet18(weights=weights_name)
        case "wide_resnet50_2":
            return torchvision.models.wide_resnet50_2(weights=weights_name)

    raise NotImplementedError("Backbone architecture not recognized!")


def load_model(model_name: str, weights_name: str) -> TorchResNet | nn.Module:
    # These are the semi supervised and weakly semi supervised weights from Facebook
    use_imagenet_1k = "swsl" in model_name or "ssl" in model_name

    if use_imagenet_1k:
        return torch.hub.load(
            "facebookresearch/semi-supervised-ImageNet1K-models",
            model_name,
        )

    return load_resnet_model(model_name, weights_name)


@dataclass
class ResNetConfig:
    """Class representing the configuration for the resnet backbone used in the pipeline
    * `pretrained` (bool, optional): Whether pretrained or not. Defaults to True.
    * `layers_to_freeze` (int, optional): The number of residual blocks to freeze (starting from 0) . Defaults to 2.
    * `crop_last_layer` (bool, optional): Whether to crop the last layer (layer 4) or not. Defaults to False.
    * `crop_second_to_last_layer` (bool, optional): Whether to crop the second to last layer (layer 3) or not. Defaults to False.
    """

    model_name: str = "resnet50"
    pretrained: bool = True
    layers_to_freeze: int = 2
    crop_last_layer: bool = False
    crop_second_to_last_layer: bool = False


class ResNet(nn.Module):
    def __init__(
        self,
        config: ResNetConfig,
    ):
        """Class representing the resnet backbone used in the pipeline
        we consider resnet network as a list of 5 blocks (from 0 to 4),
        layer 0 is the first conv+bn and the other layers (1 to 4) are the rest of the residual blocks
        we don't take into account the global pooling and the last fc

        Args:
            config (ResNetConfig): Configuration class for the resnet backbone.

        Raises:
            NotImplementedError: if the model_name corresponds to an unknown architecture.
        """
        super().__init__()
        self.__resnet_config = config
        lowercase_model_name = config.model_name.lower()

        # the new naming of pretrained weights, you can change to V2 if desired.
        weights_name = "IMAGENET1K_V1" if self.__resnet_config.pretrained else None

        # These are the semi supervised and weakly semi supervised weights from Facebook
        self.model = load_model(lowercase_model_name, weights_name)

        # freeze only if the model is pretrained
        if self.__resnet_config.pretrained:
            if self.__resnet_config.layers_to_freeze >= 0:
                self.model.conv1.requires_grad_(False)
                self.model.bn1.requires_grad_(False)

            if self.__resnet_config.layers_to_freeze >= 1:
                self.model.layer1.requires_grad_(False)

            if self.__resnet_config.layers_to_freeze >= 2:
                self.model.layer2.requires_grad_(False)

            if self.__resnet_config.layers_to_freeze >= 3:
                self.model.layer3.requires_grad_(False)

        # remove the avgpool and most importantly the fc layer
        self.model.avgpool = None
        self.model.fc = None

        use_smaller_model = (
            lowercase_model_name == "resnet34" or lowercase_model_name == "resnet18"
        )

        out_channels = 512 if use_smaller_model else 2048

        if self.__resnet_config.crop_last_layer:
            self.model.layer4 = None
            out_channels = out_channels // 2

        if self.__resnet_config.crop_second_to_last_layer:
            self.model.layer3 = None
            out_channels = out_channels // 2

        self.out_channels = out_channels

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        if self.model.layer3 is not None:
            x = self.model.layer3(x)
        if self.model.layer4 is not None:
            x = self.model.layer4(x)
        return x
