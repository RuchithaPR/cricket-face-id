"""Shared helpers: face detection (MTCNN) + embedding (FaceNet, pretrained on VGGFace2)."""
import numpy as np
import torch
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN, InceptionResnetV1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_mtcnn = MTCNN(image_size=160, margin=20, post_process=True, device=DEVICE)
_resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_image(path_or_file):
    img = Image.open(path_or_file)
    img = ImageOps.exif_transpose(img)  # fix phone-photo rotation
    return img.convert("RGB")


@torch.no_grad()
def embed(img: Image.Image):
    """Return a 512-d L2-normalised embedding of the most prominent face, or None."""
    face = _mtcnn(img)  # tensor 3x160x160 or None
    if face is None:
        return None
    emb = _resnet(face.unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
    return emb / np.linalg.norm(emb)
