"""
PhotoFinder — query app (real-time inference endpoint).

Serves a search UI. Each search embeds the text query via the SAME GPU
endpoint used at index time, then does an instant cosine search over the
local embedding index.

Run:
    uvicorn app:app --reload --port 8000
    # then open http://localhost:8000
"""

import json
import os

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

app = FastAPI(title="PhotoFinder")

_INDEX = {"emb": None, "paths": None}


def load_index():
    if _INDEX["emb"] is None:
        if not (os.path.exists("embeddings.npy") and os.path.exists("paths.json")):
            return False
        _INDEX["emb"] = np.load("embeddings.npy")
        _INDEX["paths"] = json.load(open("paths.json"))
    return True


@app.get("/")
def home():
    return HTMLResponse(open(os.path.join("static", "index.html")).read())


@app.get("/img")
def img(i: int):
    paths = _INDEX["paths"]
    if paths is None or i < 0 or i >= len(paths):
        return JSONResponse({"error": "bad index"}, status_code=404)
    return FileResponse(paths[i])


@app.get("/status")
def status():
    if not load_index():
        return {"ready": False, "count": 0}
    return {"ready": True, "count": len(_INDEX["paths"])}


@app.get("/search")
async def search(q: str, k: int = 18):
    if not load_index():
        return JSONResponse({"error": "Index not built. Run: python index.py <folder>"},
                            status_code=400)
    # Lazy import so `flash deploy` sees clip_embed defined ONLY in endpoints.py.
    from endpoints import clip_embed
    r = await clip_embed(texts=[q])
    qv = np.array(r["text_embeddings"][0], dtype=np.float32)
    sims = _INDEX["emb"] @ qv          # both sides are L2-normalized
    idx = np.argsort(-sims)[:k]
    hits = [{"i": int(i), "score": round(float(sims[i]), 3)} for i in idx]
    return JSONResponse({"device": r.get("device", "GPU"), "results": hits})


@app.get("/duplicates")
def duplicates(threshold: float = 0.93):
    if not load_index():
        return JSONResponse({"error": "Index not built."}, status_code=400)
    emb = _INDEX["emb"]
    sims = emb @ emb.T
    n = len(emb)
    seen, groups = set(), []
    for i in range(n):
        if i in seen:
            continue
        grp = [int(j) for j in range(n) if j != i and sims[i, j] >= threshold]
        if grp:
            group = [i] + grp
            groups.append([int(x) for x in group])
            seen.update(group)
    return JSONResponse(groups)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
