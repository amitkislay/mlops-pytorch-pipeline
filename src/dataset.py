"""Dataset utilities for CIFAR-10 training."""

from __future__ import annotations

import logging
import socket
import time
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

LOGGER = logging.getLogger(__name__)

CIFAR_ARCHIVE_NAME = "cifar-10-python.tar.gz"


def get_transforms(train: bool = True) -> transforms.Compose:
    """Return CIFAR-10 transformations."""

    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=CIFAR10_MEAN,
                    std=CIFAR10_STD,
                ),
            ]
        )

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=CIFAR10_MEAN,
                std=CIFAR10_STD,
            ),
        ]
    )


def remove_incomplete_archive(data_path: Path) -> None:
    """Remove a partially downloaded CIFAR-10 archive."""

    archive_path = data_path / CIFAR_ARCHIVE_NAME

    if archive_path.exists():
        LOGGER.warning(
            "Removing incomplete CIFAR-10 archive: %s",
            archive_path,
        )
        archive_path.unlink()


def download_cifar10(
    data_path: Path,
    retries: int = 5,
    retry_delay: int = 10,
) -> None:
    """Download CIFAR-10 with retry handling."""

    data_path.mkdir(parents=True, exist_ok=True)

    # Allow more time for slow connections.
    socket.setdefaulttimeout(300)

    for attempt in range(1, retries + 1):
        try:
            LOGGER.info(
                "Preparing CIFAR-10 dataset, attempt %d/%d",
                attempt,
                retries,
            )

            # Downloading one split downloads the complete CIFAR-10 archive.
            datasets.CIFAR10(
                root=str(data_path),
                train=True,
                download=True,
            )

            LOGGER.info("CIFAR-10 dataset is ready.")
            return

        except (
            ConnectionResetError,
            ConnectionError,
            TimeoutError,
            OSError,
        ) as error:
            LOGGER.warning(
                "CIFAR-10 download attempt %d failed: %s",
                attempt,
                error,
            )

            remove_incomplete_archive(data_path)

            if attempt == retries:
                raise RuntimeError(
                    "\nCIFAR-10 could not be downloaded after "
                    f"{retries} attempts.\n"
                    "Check your internet connection, VPN, proxy, "
                    "firewall, or corporate network restrictions.\n"
                    f"Expected dataset directory: {data_path.resolve()}"
                ) from error

            wait_time = retry_delay * attempt
            LOGGER.info("Retrying in %d seconds...", wait_time)
            time.sleep(wait_time)


def get_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 training and validation data loaders."""

    data_path = Path(data_dir).expanduser().resolve()

    download_cifar10(data_path=data_path)

    # download=False prevents each dataset instance from downloading again.
    train_dataset = datasets.CIFAR10(
        root=str(data_path),
        train=True,
        download=False,
        transform=get_transforms(train=True),
    )

    validation_dataset = datasets.CIFAR10(
        root=str(data_path),
        train=False,
        download=False,
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )

    return train_loader, validation_loader