import random
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


def fetch_fakeface_rest():
    resp = requests.get("https://fakeface.rest/face/json", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    url = data.get("image_url")
    if not url:
        raise RuntimeError("no image_url in fakeface response")
    img_resp = requests.get(url, timeout=10)
    img_resp.raise_for_status()
    return img_resp.content


def fetch_tpdne():
    url = f"https://thispersondoesnotexist.com/image?{random.randint(0, 1_000_000_000)}"
    resp = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "image/avif,image/webp,*/*",
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.content


def main(n=10, max_attempts=100):
    dst = Path("samples")
    dst.mkdir(exist_ok=True)
    ok = 0
    attempts = 0
    fetchers = [fetch_fakeface_rest, fetch_tpdne]

    while ok < n and attempts < max_attempts:
        attempts += 1
        for fetch in fetchers:
            try:
                content = fetch()
                img = Image.open(BytesIO(content)).convert("RGB")
                out = dst / f"fake_ai_{ok + 1}.jpg"
                img.save(out, format="JPEG")
                ok += 1
                print(f"saved {out} (attempt {attempts}, src {fetch.__name__})")
                time.sleep(0.4)
                break
            except Exception:
                continue

    print(f"downloaded {ok}/{n} images in {attempts} attempts")


if __name__ == "__main__":
    main()
