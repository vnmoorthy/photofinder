<div align="center">

# 🔎 PhotoFinder

### Search your entire photo library by *meaning* — not filenames, not tags.

Type **"people laughing at the beach"** and get the right photos back in milliseconds.
Embeddings run on **serverless GPUs** via [Runpod Flash](https://runpod.io/flash) — no Docker, scale-to-zero.

[![Live Demo](https://img.shields.io/badge/live%20demo-vnmoorthy.github.io-7c9cff?style=for-the-badge)](https://vnmoorthy.github.io/photofinder/)
[![Runpod Flash](https://img.shields.io/badge/Runpod-Flash-9b7cff?style=for-the-badge)](https://runpod.io/flash)

![Python](https://img.shields.io/badge/Python-3.10--3.13-3776AB?logo=python&logoColor=white)
![CLIP](https://img.shields.io/badge/model-CLIP%20ViT--B%2F32-46d39a)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue)

**[🌐 Live Demo](https://vnmoorthy.github.io/photofinder/)** · **[⚡ Quickstart](#-quickstart)** · **[🏗 How it works](#-how-it-works)**

</div>

---

## The 30-second pitch

Your phone has thousands of photos and the only way to find one is to scroll. Filenames are
`IMG_4821.jpg`. There are no tags. PhotoFinder fixes that: it understands what's *in* each photo,
so you can search the way you actually think — **"my dog on the beach", "that whiteboard from the
meeting", "sunset over mountains"** — and it just finds them. It also spots near-duplicates so you
can reclaim storage.

It's powered by OpenAI's **CLIP**, which embeds images and text into the *same* vector space — so a
text query can be compared directly against every photo. The heavy lifting runs on **Runpod Flash**:
GPU-backed serverless endpoints defined in pure Python, no Dockerfile, scaling from zero.

## What we built — answered plainly

| Question | Answer |
|---|---|
| **What does the workload do?** | Embeds images and text with CLIP into one shared vector space, then ranks photos against a query by cosine similarity. |
| **Where does Flash remove friction?** | No Dockerfile, no registry, no infra. A `@Endpoint`-decorated Python function *is* the GPU service. `flash deploy` → live HTTPS endpoint. Workers scale from zero and bill per-second. |
| **What input does it accept?** | A folder of images (indexing) and a natural-language query string (search). |
| **What output does it produce?** | Ranked photos with similarity scores, plus grouped near-duplicates. |
| **What becomes easier in a real product?** | Photo apps, e-commerce catalogs, DAMs, and content moderation all need "find by meaning" + dedupe. This is that core, in ~150 lines, with the GPU cost only when a job actually runs. |

## ⚡ Quickstart

> Prereqs: Python 3.10–3.13, [`uv`](https://docs.astral.sh/uv/), and a Runpod account with a verified email.

```bash
git clone https://github.com/vnmoorthy/photofinder.git
cd photofinder

uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

uv run flash login          # one-time: authorize in browser
flash deploy                # builds + deploys the GPU/CPU endpoints (first build ~5 min)

python fetch_samples.py     # OR point step below at your own photo folder
python index.py sample_images

python -m uvicorn app:app --port 8000   # open http://localhost:8000
```

> **Best demo:** `python index.py ~/Pictures` — "search *my* photos" is the moment that lands.

## 🏗 How it works

```
                 ┌──────────────────────── Runpod Flash (serverless GPU) ───────────────────────┐
                 │                                                                               │
  index.py  ─────┼──▶  clip_embed( images )   ← BATCH: fans out N parallel jobs, scales from 0   │
  (your photos)  │           │                                                                   │
                 │           ▼                                                                    │
                 │     embeddings.npy  +  paths.json   (local vector index)                       │
                 │                                                                                │
  app.py    ─────┼──▶  clip_embed( "beach at sunset" )  ← LIVE: real-time query embedding         │
  (FastAPI) ◀────┼─────────  cosine rank over the index  ──────────  ranked photos + scores       │
                 └────────────────────────────────────────────────────────────────────────────────┘
```

**One model, two roles, defined in [`endpoints.py`](endpoints.py):**
- **Index time (async batch):** `index.py` splits the library into chunks and fires them at the GPU
  endpoint *in parallel* — `clip_embed` runs on whatever GPU Flash assigns and reports its name back as proof.
- **Query time (real-time endpoint):** every search embeds the query text on the same endpoint, then a
  sub-millisecond cosine search over the local index returns the best matches.
- **Optional CPU stage:** `fetch_and_prep` (a CPU `@Endpoint`) downloads + resizes images from URLs,
  showing Flash as a true **multi-endpoint pipeline**, not just a deploy shortcut.

## 🧱 Project layout

| File | Role |
|------|------|
| [`endpoints.py`](endpoints.py) | Flash endpoints — `clip_embed` (GPU) + `fetch_and_prep` (CPU) |
| [`index.py`](index.py) | Async batch indexer — fans out embeddings across GPU workers |
| [`app.py`](app.py) | FastAPI app — live `/search`, `/duplicates`, serves the UI |
| [`static/index.html`](static/index.html) | The product UI |
| [`fetch_samples.py`](fetch_samples.py) | Grabs demo photos so you can try it in 30s |

## 🛠 Tech

**CLIP ViT-B/32** (sentence-transformers) · **Runpod Flash** serverless GPU · **FastAPI** · **NumPy** cosine retrieval · vanilla-JS UI.

## 📜 License

MIT © 2026 [vnmoorthy](https://github.com/vnmoorthy). See [LICENSE](LICENSE).

<div align="center"><sub>Built for the Runpod Flash Hack Day · powered by Runpod Flash ⚡</sub></div>
