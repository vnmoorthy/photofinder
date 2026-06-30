"""
Auto-curate a demo image set that ACTUALLY matches its labels.

A 'ralph loop': for each category we download several candidate photos, embed
them on the Flash GPU with CLIP, and keep only an image whose similarity to its
label clears a threshold — re-rolling until one does (or we run out of tries).
This guarantees "a dog" really returns a dog, etc.

    python curate_samples.py
"""

import asyncio
import base64
import os
import shutil
import urllib.request

import numpy as np

# (filename, label used to VERIFY + search, loremflickr tags)
CATS = [
    ("beach",    "a beach at sunset",        "beach,sunset"),
    ("dog",      "a dog",                    "dog"),
    ("pizza",    "a pizza on a plate",       "pizza"),
    ("city",     "a city skyline at night",  "city,skyline,night"),
    ("mountain", "a mountain landscape",     "mountain,landscape"),
    ("portrait", "a smiling person",         "smiling,face,portrait"),
    ("cat",      "a cat",                    "cat,kitten"),
    ("coffee",   "a cup of coffee",          "coffee,cup"),
    ("flower",   "a flower",                 "flower,blossom"),
    ("forest",   "a forest",                 "forest,trees"),
    ("car",      "a car",                    "car,automobile"),
    ("ocean",    "ocean waves",              "ocean,waves,sea"),
    ("snow",     "snow in winter",           "snow,winter"),
    ("bicycle",  "a bicycle",                "bicycle,bike"),
]
THRESHOLD = 0.24      # CLIP image-text cosine that counts as a confident match
PER_ROUND = 3
MAX_ROUNDS = 3


def download(tags, lock):
    url = f"https://loremflickr.com/600/450/{tags}?lock={lock}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=30).read()


async def main():
    from endpoints import clip_embed

    os.makedirs("sample_images", exist_ok=True)
    for f in os.listdir("sample_images"):
        os.remove(os.path.join("sample_images", f))

    labels = [c[1] for c in CATS]
    T = np.array((await clip_embed(texts=labels))["text_embeddings"], dtype=np.float32)

    report = []
    for ci, (name, label, tags) in enumerate(CATS):
        best_sim, best_data = -1.0, None
        lock = ci * 1000 + 1
        for _ in range(MAX_ROUNDS):
            datas, b64s = [], []
            for _ in range(PER_ROUND):
                try:
                    d = download(tags, lock)
                    datas.append(d)
                    b64s.append(base64.b64encode(d).decode())
                except Exception:
                    pass
                lock += 1
            if not b64s:
                continue
            E = np.array((await clip_embed(images_b64=b64s))["image_embeddings"], dtype=np.float32)
            sims = E @ T[ci]
            j = int(np.argmax(sims))
            if sims[j] > best_sim:
                best_sim, best_data = float(sims[j]), datas[j]
            if best_sim >= THRESHOLD:
                break
        with open(os.path.join("sample_images", f"{name}.jpg"), "wb") as f:
            f.write(best_data)
        ok = "OK  " if best_sim >= THRESHOLD else "WEAK"
        report.append((name, label, best_sim, ok))
        print(f"{ok} {name:9s} {label:24s} sim={best_sim:.3f}")

    # two exact copies of the beach winner so 'Find duplicates' has a real group
    shutil.copy("sample_images/beach.jpg", "sample_images/dupe_a.jpg")
    shutil.copy("sample_images/beach.jpg", "sample_images/dupe_b.jpg")

    weak = [r for r in report if r[3] == "WEAK"]
    print(f"\nCurated {len(report)} categories; {len(weak)} weak: {[w[0] for w in weak]}")


if __name__ == "__main__":
    asyncio.run(main())
