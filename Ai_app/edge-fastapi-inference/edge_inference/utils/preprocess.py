from PIL import Image
import io

def preprocess_image(image_bytes: bytes):
    """
    Basic preprocessing: Convert bytes to PIL Image.
    YOLOv11 handles resizing and normalization internally.
    """
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    return image
