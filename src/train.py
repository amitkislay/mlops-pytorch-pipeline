import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from dataset import get_dataloaders
from model import get_model


def log_event(**values) -> None:
    """
    Print one structured JSON event to stdout.
    """
    print(json.dumps(values), flush=True)


def load_config(config_path: str) -> dict:
    """
    Load the YAML training configuration.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Training configuration was not found: {path}"
        )

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a YAML mapping.")

    return config


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Train the model for one complete epoch.
    """
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size

        predicted_classes = outputs.argmax(dim=1)
        correct_predictions += (
            predicted_classes == targets
        ).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate the model without calculating gradients.
    """
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size

        predicted_classes = outputs.argmax(dim=1)
        correct_predictions += (
            predicted_classes == targets
        ).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    validation_accuracy: float,
    architecture: str,
    num_classes: int,
    checkpoint_path: Path,
) -> None:
    """
    Save the model and training metadata.
    """
    checkpoint = {
        "epoch": epoch,
        "architecture": architecture,
        "num_classes": num_classes,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": validation_loss,
        "val_accuracy": validation_accuracy,
    }

    torch.save(checkpoint, checkpoint_path)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a CIFAR-10 image classifier."
    )

    parser.add_argument(
        "--config",
        default=os.getenv(
            "TRAINING_CONFIG",
            "configs/training_config.yaml",
        ),
        help="Path to the YAML training configuration.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = load_config(args.config)

    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]
    output_config = config["output"]

    seed = training_config.get("seed", 42)
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    architecture = model_config["architecture"]
    num_classes = model_config["num_classes"]

    model = get_model(
        architecture=architecture,
        num_classes=num_classes,
    ).to(device)

    train_loader, validation_loader = get_dataloaders(
        data_dir=data_config["data_dir"],
        batch_size=training_config["batch_size"],
        num_workers=training_config.get("num_workers", 2),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.0),
    )

    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = Path(output_config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / output_config["model_name"]

    best_validation_loss = float("inf")
    patience_counter = 0
    patience = training_config["early_stopping_patience"]
    total_epochs = training_config["epochs"]

    log_event(
        event="training_started",
        architecture=architecture,
        device=str(device),
        epochs=total_epochs,
        batch_size=training_config["batch_size"],
        config=args.config,
    )

    for epoch in range(1, total_epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        validation_loss, validation_accuracy = evaluate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        log_event(
            event="epoch_complete",
            epoch=epoch,
            train_loss=round(train_loss, 4),
            train_accuracy=round(train_accuracy, 4),
            val_loss=round(validation_loss, 4),
            val_accuracy=round(validation_accuracy, 4),
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            patience_counter = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
                architecture=architecture,
                num_classes=num_classes,
                checkpoint_path=checkpoint_path,
            )

            log_event(
                event="checkpoint_saved",
                epoch=epoch,
                path=str(checkpoint_path),
                val_loss=round(validation_loss, 4),
            )
        else:
            patience_counter += 1

            log_event(
                event="no_improvement",
                epoch=epoch,
                patience_counter=patience_counter,
                patience_limit=patience,
            )

            if patience_counter >= patience:
                log_event(
                    event="early_stopping",
                    epoch=epoch,
                    best_val_loss=round(best_validation_loss, 4),
                )
                break

    log_event(
        event="training_complete",
        best_val_loss=round(best_validation_loss, 4),
        checkpoint_path=str(checkpoint_path),
    )


if __name__ == "__main__":
    main()
    