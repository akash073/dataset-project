import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from models.tiny_vlm_model import TinyVLM, VLMTextProcessor


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_canvas_image_for_vlm(canvas_image):
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

    preprocess = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    image_tensor = preprocess(pil_img).unsqueeze(0)
    return image_tensor


def load_tiny_vlm_model(model_path="checkpoints/mnist_tiny_vlm.pth"):
    device = get_device()
    checkpoint = torch.load(model_path, map_location=device)

    model = TinyVLM(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        text_hidden_dim=checkpoint["text_hidden_dim"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    text_processor = VLMTextProcessor()

    class_text_ids = text_processor.build_all_class_token_ids().to(device)
    with torch.inference_mode():
        cached_text_features = model.encode_text(class_text_ids)

    return model, text_processor, cached_text_features


def predict_with_tiny_vlm(canvas_image, model, cached_text_features):
    image_tensor = preprocess_canvas_image_for_vlm(canvas_image)
    if image_tensor is None:
        return None, None

    image_tensor = image_tensor.to(get_device())

    with torch.inference_mode():
        image_features = model.encode_image(image_tensor)
        logits = image_features @ cached_text_features.T
        probs = torch.softmax(logits, dim=1)

        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()

    return pred, confidence