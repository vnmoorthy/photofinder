"""
PhotoFinder — Runpod Flash endpoints.

Two endpoints, two hardware tiers (this is the "Flash is an orchestration layer" story):

  - clip_embed   GPU  : CLIP embeddings for images AND text (shared vector space)
  - fetch_and_prep CPU: optional — download + resize images from URLs (pipeline showcase)

The same GPU endpoint serves two roles:
  * BATCH (index time): called in parallel, one worker per chunk -> async fan-out
  * LIVE  (query time): called once per search -> real-time inference endpoint
"""

from runpod_flash import Endpoint, GpuGroup

# Cached across warm invocations on the same worker process.
_CACHE = {}


@Endpoint(
    name="photofinder-clip",
    gpu=GpuGroup.ANY,
    workers=5,
    idle_timeout=300,
    dependencies=["torch", "sentence-transformers", "Pillow"],
)
def clip_embed(images_b64=None, texts=None):
    """Return L2-normalized CLIP embeddings for images and/or text.

    Images and text land in the SAME 512-d space, so a text query can be
    compared directly against image vectors with a dot product.
    """
    import io
    import base64
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    if "model" not in _CACHE:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _CACHE["model"] = SentenceTransformer("clip-ViT-B-32", device=device)
        _CACHE["device"] = (
            torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
        )
    model = _CACHE["model"]

    # Echo back the real hardware this ran on — proof the Flash GPU is live.
    out = {"device": _CACHE["device"]}

    if images_b64:
        imgs = []
        for s in images_b64:
            data = base64.b64decode(s)
            imgs.append(Image.open(io.BytesIO(data)).convert("RGB"))
        emb = model.encode(imgs, batch_size=16, normalize_embeddings=True)
        out["image_embeddings"] = emb.tolist()

    if texts:
        emb = model.encode(texts, normalize_embeddings=True)
        out["text_embeddings"] = emb.tolist()

    return out


@Endpoint(
    name="photofinder-prep",
    cpu="cpu5c-4-8",
    idle_timeout=300,
    dependencies=["Pillow", "requests"],
)
def fetch_and_prep(urls):
    """CPU pre-processing stage: download + downscale images from URLs.

    Optional — only used when you index from a list of URLs instead of a
    local folder. Demonstrates the CPU -> GPU multi-endpoint pipeline.
    """
    import io
    import base64
    import requests
    from PIL import Image

    out = []
    for u in urls:
        resp = requests.get(u, timeout=20)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img.thumbnail((512, 512))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        out.append(base64.b64encode(buf.getvalue()).decode())
    return {"images_b64": out}
