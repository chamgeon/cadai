"""Step 4: embed every rendered view with DINOv2 and CLIP.

Each view gets its own vector; how the 6 views become one score (mean, max,
or multi-vector MaxSim) is decided at search time, so it can be compared
without re-running the models.

Output: data/embeddings/{render_set}__{model}.npz with
    part_ids  (N,)        vectors  (N, 6, D), L2-normalized per view

Usage:
    python -m src.embed                          # both models, canonical renders
    python -m src.embed --renders renders_world  # the pre-canonicalization renders
    python -m src.embed --model dinov2
"""
import argparse
import time

import numpy as np
import torch
from PIL import Image

from src.config import DATA, RENDERS
from src.render import VIEWS

EMBEDDINGS = DATA / "embeddings"
MODELS = {
    "dinov2": "facebook/dinov2-base",  # self-supervised on images: what things look like
    "clip": "openai/clip-vit-base-patch16",  # trained on image-caption pairs: what things are called
}
BATCH = 32


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Embedder:
    def __init__(self, name):
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

        self.name = name
        if name == "dinov2":
            self.processor = AutoImageProcessor.from_pretrained(MODELS[name])
            self.model = AutoModel.from_pretrained(MODELS[name])
        else:
            self.processor = CLIPProcessor.from_pretrained(MODELS[name]).image_processor
            self.model = CLIPModel.from_pretrained(MODELS[name])
        self.model = self.model.eval().to(device())

    @torch.no_grad()
    def __call__(self, images):
        """PIL images or image paths -> (n, D) L2-normalized float32. Paths are opened per batch."""
        out = []
        for i in range(0, len(images), BATCH):
            batch = [(im if isinstance(im, Image.Image) else Image.open(im)).convert("RGB")
                     for im in images[i : i + BATCH]]
            if self.name == "dinov2":
                # The default resize-256-then-crop-224 cuts ~6% off every edge, part outlines included.
                x = self.processor(images=batch, return_tensors="pt", do_center_crop=False,
                                   size={"height": 224, "width": 224})["pixel_values"]
                v = self.model(pixel_values=x.to(device())).pooler_output  # CLS token after layer norm
            else:
                x = self.processor(images=batch, return_tensors="pt")["pixel_values"]  # square: crop is a no-op
                v = self.model.get_image_features(pixel_values=x.to(device())).pooler_output  # projected, 512-d
            out.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)


def embed_dir(embedder, render_dir, part_ids=None):
    """Embed {render_dir}/{part_id}/view_0..5.png -> (part_ids, (N, 6, D))."""
    if part_ids is None:
        part_ids = sorted(p.name for p in render_dir.iterdir() if (p / "view_0.png").exists())
    images = [render_dir / pid / f"view_{i}.png" for pid in part_ids for i in range(len(VIEWS))]
    vecs = embedder(images)
    return np.array(part_ids), vecs.reshape(len(part_ids), len(VIEWS), -1)


def embed_part(embedder, part_id):
    """One part's renders/{part_id}/view_0..5.png -> (6, D)."""
    return embed_dir(embedder, RENDERS, [part_id])[1][0]


def load(render_set, model):
    data = np.load(EMBEDDINGS / f"{render_set}__{model}.npz")
    return data["part_ids"], data["vectors"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders", default="renders", help="folder under data/ holding {part_id}/view_k.png")
    parser.add_argument("--model", choices=list(MODELS), help="default: all")
    args = parser.parse_args()

    EMBEDDINGS.mkdir(exist_ok=True)
    for name in [args.model] if args.model else MODELS:
        start = time.perf_counter()
        embedder = Embedder(name)
        loaded = time.perf_counter()
        part_ids, vecs = embed_dir(embedder, DATA / args.renders)
        out = EMBEDDINGS / f"{args.renders}__{name}.npz"
        np.savez(out, part_ids=part_ids, vectors=vecs)
        n_img = vecs.shape[0] * vecs.shape[1]
        took = time.perf_counter() - loaded
        print(f"{name:7} {vecs.shape[0]} parts x {vecs.shape[1]} views -> dim {vecs.shape[2]}  "
              f"model load {loaded - start:.1f}s, embed {took:.1f}s ({n_img / took:.0f} img/s)  -> {out.name}")


if __name__ == "__main__":
    main()
