#!/usr/bin/env python3
"""Regenerate the review previews in docs/ from 8k/ and the source JPEGs.

Two artefacts, both deliberately small enough to render inline on GitHub:

  docs/preview-contact-sheet.jpg  every 8K result, thumbnailed
  docs/preview-detail-1to1.jpg    one region at 1:1, source vs lanczos vs AI

Usage:  .venv/bin/python scripts/make_previews.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
EIGHTK = ROOT / "8k"
DOCS = ROOT / "docs"

BG = (24, 24, 28)
FG = (238, 238, 242)


def font(size: int) -> ImageFont.ImageFont:
    """Default bitmap font at a usable size (Pillow >= 10.1), else fallback."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def source_for(result: Path) -> Path:
    """8k/<stem>_<W>x<H>.jpg -> ./<stem>.jpeg"""
    return ROOT / (result.stem.rsplit("_", 1)[0] + ".jpeg")


def contact_sheet(cols: int = 4, cell: tuple[int, int] = (400, 300)) -> None:
    files = sorted(EIGHTK.glob("*.jpg"))
    if not files:
        raise SystemExit("no images in 8k/ -- nothing to preview")

    cw, ch = cell
    label_h = 30
    rows = -(-len(files) // cols)
    sheet = Image.new("RGB", (cw * cols, (ch + label_h) * rows), BG)
    draw = ImageDraw.Draw(sheet)
    f = font(20)

    for i, file in enumerate(files):
        with Image.open(file) as im:
            im.draft("RGB", (cw * 2, ch * 2))  # fast partial JPEG decode
            im = im.convert("RGB")
            im.thumbnail((cw - 8, ch - 8))
        x = (i % cols) * cw
        y = (i // cols) * (ch + label_h)
        sheet.paste(im, (x + (cw - im.width) // 2, y + (ch - im.height) // 2))
        w, h = file.stem.rsplit("_", 1)[1].split("x")
        draw.text((x + 6, y + ch + 4), f"{w} x {h}", fill=FG, font=f)

    DOCS.mkdir(exist_ok=True)
    out = DOCS / "preview-contact-sheet.jpg"
    sheet.save(out, "JPEG", quality=86, optimize=True, progressive=True)
    print(f"wrote {out.relative_to(ROOT)}  {sheet.size}  {out.stat().st_size / 1e3:.0f}KB")


def detail_1to1(
    result_name: str = "WhatsApp Image 2026-09-14 at 15.21.12 (1)_7680x4272.jpg",
    region: tuple[int, int, int, int] = (990, 290, 200, 145),
) -> None:
    """Crop one region at true 1:1 output pixels so the comparison is honest.

    `region` is in *source* coordinates; it is scaled up for the 8K crop.
    """
    result = EIGHTK / result_name
    src_path = source_for(result)
    if not (result.exists() and src_path.exists()):
        raise SystemExit(f"missing {result.name} or {src_path.name}")

    with Image.open(result) as r:
        scale = r.width / Image.open(src_path).width
        cx, cy, cw, ch = region
        box = (int(cx * scale), int(cy * scale),
               int((cx + cw) * scale), int((cy + ch) * scale))
        ai = r.crop(box).convert("RGB")

    src_crop = Image.open(src_path).convert("RGB").crop((cx, cy, cx + cw, cy + ch))
    size = ai.size

    panels = [
        (f"source, nearest {scale:.0f}x", src_crop.resize(size, Image.NEAREST)),
        (f"lanczos {scale:.0f}x", src_crop.resize(size, Image.LANCZOS)),
        ("Real-ESRGAN x4plus (delivered)", ai),
    ]

    # Labels are sized relative to the panel so they survive the final downscale.
    W, H = size
    pad = 12
    label_h = max(34, W // 26)
    f = font(max(18, W // 34))
    out_img = Image.new("RGB", (W * 3 + pad * 4, H + label_h + pad * 2), BG)
    draw = ImageDraw.Draw(out_img)
    for i, (name, im) in enumerate(panels):
        x = pad + i * (W + pad)
        out_img.paste(im, (x, label_h + pad))
        draw.text((x + 4, pad), name, fill=FG, font=f)

    out_img.thumbnail((2400, 2400), Image.LANCZOS)
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "preview-detail-1to1.jpg"
    out_img.save(out, "JPEG", quality=90, optimize=True, progressive=True)
    print(f"wrote {out.relative_to(ROOT)}  {out_img.size}  {out.stat().st_size / 1e3:.0f}KB")


if __name__ == "__main__":
    contact_sheet()
    detail_1to1()
