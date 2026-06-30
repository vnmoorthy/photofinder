"""
Bake the static live-demo site (docs/) from REAL CLIP results.

Embeds a set of example queries via the live Flash GPU endpoint, ranks them
against the indexed photos, copies the images into docs/images/, and writes
docs/demo-data.json. The GitHub Pages site then replays these real results
client-side — a genuine demo with no backend or API keys exposed.

    python build_demo.py        # after index.py has produced embeddings.npy
"""

import asyncio
import json
import os
import shutil

import numpy as np

EXAMPLES = [
    "a dog", "a cat", "beach at sunset", "mountains", "city at night",
    "food on a plate", "flowers", "a car", "coffee", "a forest",
    "snow", "someone smiling",
]
TOPK = 8
DUPE_THRESHOLD = 0.93
REL_MARGIN = 0.04
REL_FLOOR = 0.21


async def main():
    from endpoints import clip_embed  # lazy import (keeps flash deploy happy)

    emb = np.load("embeddings.npy")
    paths = json.load(open("paths.json"))

    os.makedirs("docs/images", exist_ok=True)
    files = []
    for i, p in enumerate(paths):
        ext = os.path.splitext(p)[1] or ".jpg"
        fn = f"img{i}{ext}"
        shutil.copy(p, os.path.join("docs/images", fn))
        files.append(fn)

    r = await clip_embed(texts=EXAMPLES)
    device = r.get("device", "GPU")
    tvecs = np.array(r["text_embeddings"], dtype=np.float32)

    queries = {}
    for q, tv in zip(EXAMPLES, tvecs):
        sims = emb @ tv
        idx = np.argsort(-sims)[:TOPK]
        hits = [{"file": files[i], "score": round(float(sims[i]), 3)} for i in idx]
        if hits:
            cutoff = max(REL_FLOOR, hits[0]["score"] - REL_MARGIN)
            hits = [h for h in hits if h["score"] >= cutoff]
        queries[q] = hits

    sims = emb @ emb.T
    n = len(emb)
    seen, groups = set(), []
    for i in range(n):
        if i in seen:
            continue
        grp = [int(j) for j in range(n) if j != i and sims[i, j] >= DUPE_THRESHOLD]
        if grp:
            g = [i] + grp
            groups.append([files[k] for k in g])
            seen.update(g)

    data = {
        "indexed": len(files),
        "device": device,
        "topk": TOPK,
        "files": files,
        "queries": queries,
        "duplicates": groups,
        # rounded image embeddings → client-side "find similar" on the static site
        "embeddings": np.round(emb, 4).tolist(),
    }
    with open("docs/demo-data.json", "w") as f:
        json.dump(data, f)
    print(f"Wrote docs/demo-data.json — {len(files)} images, "
          f"{len(queries)} queries, {len(groups)} dupe group(s), embedded on {device}")


if __name__ == "__main__":
    asyncio.run(main())
