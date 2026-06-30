"""
PhotoFinder — indexer (async batch fan-out).

Reads a local folder of photos, encodes them locally to base64, then fans the
work out to many GPU workers in parallel via the clip_embed endpoint.
Saves embeddings.npy + paths.json for the query app to use.

Usage:
    python index.py <folder>          # default: sample_images
"""

import asyncio
import base64
import glob
import io
import json
import os
import sys
import time

import numpy as np
from PIL import Image

CHUNK = 16          # images per GPU worker call
MAX_SIDE = 512      # downscale before upload


def encode_image_file(path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def find_images(folder):
    exts = ("jpg", "jpeg", "png", "webp", "bmp", "gif")
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(folder, f"**/*.{ext}"), recursive=True)
        paths += glob.glob(os.path.join(folder, f"**/*.{ext.upper()}"), recursive=True)
    return sorted(set(paths))


async def main(folder):
    # Lazy import so `flash deploy` sees clip_embed defined ONLY in endpoints.py.
    from endpoints import clip_embed
    paths = find_images(folder)
    if not paths:
        print(f"No images found in '{folder}'. Drop some photos there first.")
        return

    print(f"Found {len(paths)} images. Encoding locally...")
    b64s = [encode_image_file(p) for p in paths]

    chunks = [b64s[i:i + CHUNK] for i in range(0, len(b64s), CHUNK)]
    print(f"Fanning out {len(chunks)} parallel jobs to Flash GPU workers (scaling from zero)...")

    t0 = time.time()
    results = await asyncio.gather(*[clip_embed(images_b64=c) for c in chunks])
    dt = time.time() - t0

    embs = []
    for r in results:
        embs.extend(r["image_embeddings"])
    embs = np.array(embs, dtype=np.float32)

    np.save("embeddings.npy", embs)
    with open("paths.json", "w") as f:
        json.dump(paths, f)

    gpu = results[0].get("device", "GPU") if results else "GPU"
    print(f"Indexed {len(paths)} images in {dt:.1f}s "
          f"({len(paths) / dt:.0f} img/s) on {gpu} "
          f"across {len(chunks)} parallel Flash jobs -> embeddings.npy")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "sample_images"
    asyncio.run(main(folder))
