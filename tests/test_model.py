import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from model import get_model


def test_resnet18_output_shape():
    model = get_model(
        architecture="resnet18",
        num_classes=10,
    )

    model.eval()

    sample_batch = torch.randn(
        4,
        3,
        32,
        32,
    )

    with torch.no_grad():
        output = model(sample_batch)

    assert output.shape == (4, 10)


def test_model_supports_different_class_count():
    model = get_model(
        architecture="resnet18",
        num_classes=5,
    )

    model.eval()

    sample_batch = torch.randn(
        2,
        3,
        32,
        32,
    )

    with torch.no_grad():
        output = model(sample_batch)

    assert output.shape == (2, 5)


def test_invalid_architecture_raises_error():
    with pytest.raises(
        ValueError,
        match="Unsupported architecture",
    ):
        get_model(
            architecture="unknown-model",
            num_classes=10,
        )