
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


def build_model(
    name: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = True,
):
    """
    Build and return a classification model.

    Args:
        name: Model architecture name
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
    """

    if name == "resnet18":
        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )
        model = models.resnet18(weights=weights)
    else:
        raise ValueError(f"Unsupported model: {name}")

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model
