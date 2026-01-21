import torch.nn as nn
import torchvision.models as models


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
        model = models.resnet18(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model: {name}")

    # Replace classification head
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model
