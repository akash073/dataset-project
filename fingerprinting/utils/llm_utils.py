import numpy as np
import torch
from PIL import Image, ImageOps

from models.tiny_llm_model import TinyLLMClassifier, MNISTTokenDataset


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_canvas_image_for_llm(canvas_image):
    if canvas_image is None:
        return None

    image = canvas_image

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.ndim == 3:
        pil_img = Image.fromarray(image).convert("L")
    else:
        pil_img = Image.fromarray(image).convert("L")

    pil_img = ImageOps.invert(pil_img)
    pil_img = pil_img.resize((28, 28))

    img_array = np.array(pil_img).astype(np.float32) / 255.0
    image_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)
    return image_tensor


def load_tiny_llm_model(model_path="checkpoints/mnist_tiny_llm.pth"):
    device = get_device()
    checkpoint = torch.load(model_path, map_location=device)

    model = TinyLLMClassifier(
        vocab_size=checkpoint["vocab_size"],
        seq_len=checkpoint["seq_len"],
        embed_dim=checkpoint["embed_dim"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
        ff_dim=checkpoint["ff_dim"],
        num_classes=checkpoint["num_classes"],
        dropout=0.1
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    helper_dataset = MNISTTokenDataset(train=False, root="./data")
    return model, helper_dataset


def predict_with_tiny_llm(canvas_image, model, helper_dataset):
    image_tensor = preprocess_canvas_image_for_llm(canvas_image)

    if image_tensor is None:
        return None, None

    input_ids = helper_dataset.build_input_ids(image_tensor).unsqueeze(0)
    input_ids = input_ids.to(get_device())

    with torch.inference_mode():
        logits = model(input_ids)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()

    return pred, confidence