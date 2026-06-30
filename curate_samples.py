"""
Auto-curate a RICH demo image set that actually matches its labels.

A 'ralph loop': for each category we download many candidate photos, embed them
on the Flash GPU with CLIP, and keep the TOP-N whose similarity to the label
clears a threshold — re-rolling until we have enough (or run out of tries).
Multiple images per concept means "a dog" returns a grid of dogs, not one dog
plus noise.

    python curate_samples.py
"""

import asyncio
import base64
import os
import shutil
import urllib.request

import numpy as np

# (base name, label to VERIFY against, loremflickr tags)
CATS = [
    ("dog",      "a photo of a dog",                "dog"),
    ("cat",      "a photo of a cat",                "cat,kitten"),
    ("beach",    "a beach at sunset",               "beach,sunset"),
    ("mountain", "a mountain landscape",            "mountain,landscape"),
    ("city",     "a city skyline at night",         "city,skyline,night"),
    ("pizza",    "a pizza or food on a plate",      "pizza,food"),
    ("flower",   "a flower",                        "flower,blossom"),
    ("car",      "a car",                           "car,automobile"),
    ("coffee",   "a cup of coffee",                 "coffee,cup"),
    ("forest",   "a forest with trees",             "forest,trees"),
    ("snow",     "snow in winter",                  "snow,winter"),
    ("portrait", "a smiling person's face",         "smiling,face,portrait"),
]
N_PER = 3            # keep this many verified images per category
THRESHOLD = 0.23     # CLIP image-text cosine that counts as a confident match
PER_ROUND = 5
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
        cands = []  # (sim, bytes)
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
            if b64s:
                E = np.array((await clip_embed(images_b64=b64s))["image_embeddings"], dtype=np.float32)
                sims = E @ T[ci]
                cands += list(zip(sims.tolist(), datas))
            strong = [c for c in cands if c[0] >= THRESHOLD]
            if len(strong) >= N_PER:
                break
        cands.sort(key=lambda c: -c[0])
        keep = cands[:N_PER]
        for n, (sim, data) in enumerate(keep, 1):
            with open(os.path.join("sample_images", f"{name}{n}.jpg"), "wb") as f:
                f.write(data)
        sims_kept = [round(s, 3) for s, _ in keep]
        weak = any(s < THRESHOLD for s, _ in keep)
        report.append((name, sims_kept, weak))
        print(f"{'WEAK' if weak else 'OK  '} {name:9s} kept={len(keep)} sims={sims_kept}")

    # exact-duplicate pair so 'Find duplicates' has a real group
    if os.path.exists("sample_images/beach1.jpg"):
        shutil.copy("sample_images/beach1.jpg", "sample_images/dupe_a.jpg")
        shutil.copy("sample_images/beach1.jpg", "sample_images/dupe_b.jpg")

    total = len(os.listdir("sample_images"))
    weak = [r[0] for r in report if r[2]]
    print(f"\nCurated {total} images across {len(CATS)} categories; weak: {weak or 'none'}")


if __name__ == "__main__":
    asyncio.run(main())
