import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from dataset import CIFAR10_MEAN, CIFAR10_STD
from model import get_model


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        "checkpoints/classifier_v1.pt",
    )
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model: torch.nn.Module | None = None

inference_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=CIFAR10_MEAN,
            std=CIFAR10_STD,
        ),
    ]
)


def load_model(checkpoint_path: Path) -> torch.nn.Module:
    """
    Load a model from its saved checkpoint.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint was not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    architecture = checkpoint.get("architecture", "resnet18")
    num_classes = checkpoint.get("num_classes", 10)

    loaded_model = get_model(
        architecture=architecture,
        num_classes=num_classes,
    )

    loaded_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    loaded_model.to(device)
    loaded_model.eval()

    return loaded_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model when the FastAPI application starts.
    """
    global model

    try:
        model = load_model(MODEL_PATH)
    except Exception as error:
        model = None
        print(
            f"Model loading failed: {error}",
            flush=True,
        )

    yield

    model = None


app = FastAPI(
    title="CIFAR-10 Classification API",
    description="Serve predictions from a trained PyTorch model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """
    Return 200 only when the model is loaded.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device),
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
) -> dict:
    """
    Classify an uploaded image and return class probabilities.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    allowed_types = {
        "image/jpeg",
        "image/png",
    }

    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only PNG and JPEG images are supported.",
        )

    try:
        image_bytes = await image.read()
        pil_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        ) from error

    input_tensor = inference_transform(
        pil_image
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    predicted_index = int(
        probabilities.argmax().item()
    )

    class_probabilities = {
        class_name: round(
            float(probabilities[index].item()),
            6,
        )
        for index, class_name in enumerate(CLASS_NAMES)
    }

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "predicted_class_index": predicted_index,
        "confidence": round(
            float(probabilities[predicted_index].item()),
            6,
        ),
        "probabilities": class_probabilities,
    }
