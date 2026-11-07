import subprocess, sys
try:
    import facenet_pytorch
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "facenet-pytorch==2.6.0", "--no-deps", "-q"])

import joblib
import numpy as np
import torch
import gradio as gr
import spaces
from PIL import Image, ImageOps, ImageDraw, ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face

mtcnn = MTCNN(device="cpu")                      # used for detection only
resnet = InceptionResnetV1(pretrained="vggface2").eval()
clf = joblib.load("model.joblib")

NAMES = {"ms_dhoni": "MS Dhoni", "k._l._rahul": "KL Rahul"}
MAX_FACES = 10


def pretty(n):
    return NAMES.get(n, " ".join(n.split("_")).title())


def get_font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


@spaces.GPU
@torch.no_grad()
def identify(img, threshold):
    if img is None:
        raise gr.Error("Please upload a photo.")
    img = ImageOps.exif_transpose(img).convert("RGB")

    boxes, probs = mtcnn.detect(img)
    if boxes is None:
        raise gr.Error("No face detected. Try a clearer photo.")
    keep = [i for i, p in enumerate(probs) if p is not None and p >= 0.9][:MAX_FACES]
    if not keep:
        raise gr.Error("No clear face detected. Try a clearer, larger face.")
    boxes = boxes[keep]
    order = np.argsort(boxes[:, 0])               # number faces left to right
    boxes = boxes[order]

    faces = torch.stack([extract_face(img, b, image_size=160, margin=20) for b in boxes])
    emb = resnet((faces - 127.5) / 128.0).numpy()
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    proba = clf.predict_proba(emb)

    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = get_font(max(14, img.width // 45))
    lines = []
    for n, (box, p) in enumerate(zip(boxes, proba), 1):
        top = np.argsort(p)[::-1][:3]
        best, conf = pretty(clf.classes_[top[0]]), float(p[top[0]])
        known = conf >= threshold
        label = f"{best} {conf:.0%}" if known else f"Unknown ({conf:.0%})"
        color = (30, 200, 80) if known else (230, 60, 60)
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(3, img.width // 300))
        ty = max(0, y1 - font.size - 6) if hasattr(font, "size") else max(0, y1 - 20)
        tw = draw.textlength(label, font=font)
        draw.rectangle([x1, ty, x1 + tw + 8, ty + (font.size + 6 if hasattr(font, "size") else 20)], fill=color)
        draw.text((x1 + 4, ty + 2), label, fill=(255, 255, 255), font=font)
        alts = ", ".join(f"{pretty(clf.classes_[i])} {p[i]:.0%}" for i in top)
        lines.append(f"**Face {n}:** {label}  \n<sub>Top 3: {alts}</sub>")
    return out, "\n\n".join(lines)


demo = gr.Interface(
    fn=identify,
    inputs=[
        gr.Image(type="pil", label="Upload a photo (one or many faces)"),
        gr.Slider(0.1, 0.95, value=0.5, label="Confidence threshold (below this = Unknown)"),
    ],
    outputs=[gr.Image(label="Detected faces"), gr.Markdown(label="Results")],
    title="🏏 Indian Cricketer Face Identifier",
    description="Detects every face in a photo and identifies it among 15 Indian cricket team members. "
                "Pipeline: MTCNN detection → FaceNet embeddings → classifier.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
