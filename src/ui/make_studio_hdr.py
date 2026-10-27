"""Step 7: write src/ui/static/studio.hdr, the lighting of the 3D view.

model-viewer lights a model only by an environment image. Its built-in ones light
from everywhere at once, so the faces of a machined part all came out the same light
grey and the view looked washed out next to the renders (VTK light kit: a key light
from one side). This one is a dim room with a strong key light up front and a weaker
fill behind: faces now differ by the direction they face. It has to be HDR: an
8-bit PNG can't make the key light brighter than white, and the blurring model-viewer
applies for diffuse light flattens it again.

    python -m src.ui.make_studio_hdr
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "static" / "studio.hdr"
KEY, FILL, AMBIENT, GROUND = 25.0, 5.0, 0.3, 0.4  # radiance; GROUND scales everything below the horizon
KEY_DIR, FILL_DIR = (35, 50), (-130, 20)  # (azimuth, elevation) in degrees, glTF axes (y up)
KEY_SHARPNESS, FILL_SHARPNESS = 40, 16  # cosine-lobe exponents: higher is a smaller light


def lobe(dirs, azimuth, elevation, power):
    az, el = np.radians(azimuth), np.radians(elevation)
    to_light = np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
    return np.clip(dirs @ to_light, 0, None) ** power


def studio(width=256, height=128):
    """Equirectangular radiance map, grey."""
    lon = (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi
    lon, lat = np.meshgrid(lon, lat)
    dirs = np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], -1)
    img = AMBIENT + KEY * lobe(dirs, *KEY_DIR, KEY_SHARPNESS) + FILL * lobe(dirs, *FILL_DIR, FILL_SHARPNESS)
    return np.where(dirs[..., 1] < 0, img * GROUND, img)


def write_hdr(path, img):
    """Radiance RGBE with flat (not run-length encoded) scanlines; grey, so R = G = B."""
    mantissa, exponent = np.frexp(img)
    rgbe = np.zeros(img.shape + (4,), np.uint8)
    rgbe[..., :3] = (mantissa * 256).astype(np.uint8)[..., None]
    rgbe[..., 3] = exponent + 128
    with open(path, "wb") as f:
        f.write(f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {img.shape[0]} +X {img.shape[1]}\n".encode())
        f.write(rgbe.tobytes())


if __name__ == "__main__":
    write_hdr(OUT, studio())
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
