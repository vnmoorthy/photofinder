"""
Grab a few recognizable demo photos so semantic search visibly works.

Uses loremflickr (keyword-matched Creative-Commons Flickr images), so "beach"
really returns a beach, "dog" a dog, etc. `lock` pins a specific image so the
set is reproducible.

BEST DEMO: point index.py at your OWN photo folder instead — "search MY photos"
is the story that lands.

    python fetch_samples.py
"""

import os
import urllib.request

BASE = "https://loremflickr.com/600/450"

# (filename, comma-separated keywords, lock-id-for-reproducibility)
SPEC = [
    ("beach.jpg",      "beach,sunset",      11),
    ("dog.jpg",        "dog",               12),
    ("pizza.jpg",      "pizza,plate",       13),
    ("city_night.jpg", "city,night",        14),
    ("mountain.jpg",   "mountain,landscape",15),
    ("portrait.jpg",   "portrait,smile",    16),
    ("cat.jpg",        "cat",               17),
    ("coffee.jpg",     "coffee,cup",        18),
    ("flower.jpg",     "flower",            19),
    ("forest.jpg",     "forest,trees",      20),
    ("car.jpg",        "car",               21),
    ("ocean.jpg",      "ocean,waves",       22),
    ("snow.jpg",       "snow,winter",       23),
    ("bicycle.jpg",    "bicycle",           24),
    ("street.jpg",     "street,city",       25),
    ("food.jpg",       "burger,food",       26),
    # two identical pulls (same lock) so "Find duplicates" has something to find:
    ("dupe_a.jpg",     "sunset,beach",      99),
    ("dupe_b.jpg",     "sunset,beach",      99),
]


def main():
    os.makedirs("sample_images", exist_ok=True)
    for name, tags, lock in SPEC:
        url = f"{BASE}/{tags}?lock={lock}"
        dest = os.path.join("sample_images", name)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r, open(dest, "wb") as f:
                f.write(r.read())
            print("ok  ", dest)
        except Exception as e:
            print("skip", name, e)
    print("\nTip: for a real wow demo, run:  python index.py /path/to/your/photos")


# Guard so `flash deploy` (which imports every .py to find endpoints) doesn't
# re-download on every deploy.
if __name__ == "__main__":
    main()
