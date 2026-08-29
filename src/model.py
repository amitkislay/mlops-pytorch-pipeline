import torch.nn as nn
from torchvision import models


SUPPORTED_ARCHITECTURES = ("resnet18",)


def get_model(
    architecture: str = "resnet18",
    num_classes: int = 10,
) -> nn.Module:
    """
    Create an image-classification model.

    Args:
        architecture: Name of the model architecture.
        num_classes: Number of output classes.

    Returns:
        Configured PyTorch model.

    Raises:
        ValueError: If the requested architecture is unsupported.
    """
    architecture = architecture.lower()

    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture: {architecture}. "
            f"Supported architectures: {SUPPORTED_ARCHITECTURES}"
        )

    model = models.resnet18(weights=None)

    # CIFAR-10 images are 32 x 32, so use a smaller first convolution.
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    # The original ResNet max-pooling layer reduces CIFAR-10 images
    # too aggressively.
    model.maxpool = nn.Identity()

    # Replace the original classification layer.
    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes,
    )

    return model