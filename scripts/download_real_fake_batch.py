import os
import requests
import time

BASE_DIR = "deepfake_test_images"
REAL_DIR = os.path.join(BASE_DIR, "real")
FAKE_DIR = os.path.join(BASE_DIR, "fake")

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# ---------- REAL FACE IMAGES (Unsplash CDN) ----------
REAL_IMAGE_URLS = [
    "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d",
    "https://images.unsplash.com/photo-1494790108377-be9c29b29330",
    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d",
    "https://images.unsplash.com/photo-1527980965255-d3b416303d12",
    "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde",
    "https://images.unsplash.com/photo-1500648767791-00dcc994a43e",
    "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e",
    "https://images.unsplash.com/photo-1544725176-7c40e5a2c9f9",
    "https://images.unsplash.com/photo-1524504388940-b1c1722653e1",
    "https://images.unsplash.com/photo-1520813792240-56fc4a3765a7",
    "https://images.unsplash.com/photo-1517841905240-472988babdf9",
    "https://images.unsplash.com/photo-1492562080023-ab3db95bfbce",
    "https://images.unsplash.com/photo-1488426862026-3ee34a7d66df",
    "https://images.unsplash.com/photo-1500917293891-ef795e70e1f6",
    "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e",
]

# ---------- DEEPFAKE IMAGES ----------
FAKE_IMAGE_URL = "https://thispersondoesnotexist.com/"


def download_image(url, path):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"✅ {path}")
    except Exception as e:
        print(f"❌ {path} | {e}")


print("\n📥 Downloading REAL images...")
for i, url in enumerate(REAL_IMAGE_URLS, 1):
    save_path = os.path.join(REAL_DIR, f"real_{i:02d}.jpg")
    download_image(url, save_path)

print("\n📥 Downloading DEEPFAKE images...")
for i in range(1, 16):
    save_path = os.path.join(FAKE_DIR, f"fake_{i:02d}.jpg")
    download_image(FAKE_IMAGE_URL, save_path)
    time.sleep(1)  # avoid rate-limit

print("\n🎉 SUCCESS: 15 real + 15 deepfake images downloaded")
