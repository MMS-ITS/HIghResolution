#!/usr/bin/env python3
"""
Upscale images to 8K (or any target) using Real-ESRGAN via ONNX Runtime.

The pipeline is: AI super-resolution in 4x steps until the image is at least as
large as the target, then a high-quality Lanczos step down to the exact target
size. Downsampling from an oversized AI result is what produces the crisp final
image -- resizing straight to the target would leave it soft.

Examples
--------
    # Everything in ./input -> 8K (7680px long edge) in ./output
    python upscale.py

    # Explicit paths, fast model
    python upscale.py -i photos/ -o out/ --method fast

    # Exact 7680x4320 canvas, crop to fill
    python upscale.py --exact 7680x4320 --fit cover

    # See the plan without doing the work
    python upscale.py --dry-run
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageFilter

# Large intermediates are intentional here, not a decompression bomb.
Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"

# name -> (filename, scale, min_overlap, description)
#
# `min_overlap` is the tile overlap margin needed to make tiled inference match
# whole-image inference, and it is a property of the model's receptive field --
# not a free tuning knob. Measured against whole-image output on a photo:
#
#   quality  overlap 16 -> max error 92/255 (obvious seams)
#            overlap 64 -> 9, overlap 96 -> 1
#   fast     overlap 16 -> max error  4/255
#            overlap 32 -> 1
#
# RRDBNet-23 aggregates over a much wider area than SRVGGNetCompact, hence the
# large difference. Do not lower these without re-running tests/test_upscale.py.
MODELS: dict[str, tuple[str, int, int, str]] = {
    "quality": (
        "realesrgan_x4plus.onnx",
        4,
        96,
        "Real-ESRGAN x4plus (RRDBNet-23, 64MB) - best detail, slowest",
    ),
    "fast": (
        "realesr-general-x4v3.onnx",
        4,
        32,
        "Real-ESRGAN general x4v3 (SRVGGNetCompact, 5MB) - ~15x faster, slightly softer",
    ),
}

# Named targets, expressed as the long edge in pixels.
TARGETS: dict[str, int] = {
    "8k": 7680,      # UHD-2 width; gives 7680x4320 on 16:9 sources
    "8k-dci": 8192,  # cinema / DCI convention
    "6k": 6144,
    "5k": 5120,
    "4k": 3840,
    "4k-dci": 4096,
}

READ_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".ppm", ".pgm"}

# Beyond this, warn the user that RAM is about to get interesting.
INTERMEDIATE_PIXEL_WARN = 300_000_000


# --------------------------------------------------------------------------- #
# Model inference
# --------------------------------------------------------------------------- #


class OnnxUpscaler:
    """Tiled 4x super-resolution using an ONNX Real-ESRGAN model.

    Tiles are read with an overlap margin and the margin is discarded from the
    output. Because every output pixel comes from a tile where it sat at least
    `overlap` pixels from the edge, the result is free of the seams you get from
    naive tiling -- no blending needed.
    """

    def __init__(
        self,
        model_path: Path,
        scale: int = 4,
        tile: int = 256,
        overlap: int = 16,
        threads: int | None = None,
    ) -> None:
        import onnxruntime as ort  # imported lazily so --method lanczos needs no ORT

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or (os.cpu_count() or 4)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.scale = scale
        self.tile = tile
        self.overlap = overlap

    def _infer(self, patch: np.ndarray) -> np.ndarray:
        """patch: float32 HWC in [0,1] -> float32 HWC upscaled."""
        x = np.ascontiguousarray(patch.transpose(2, 0, 1)[None])
        y = self.session.run(None, {self.input_name: x})[0]
        return y[0].transpose(1, 2, 0)

    def upscale(self, img: np.ndarray, on_tile=None) -> np.ndarray:
        """img: uint8 HWC RGB -> uint8 HWC RGB, `scale`x larger."""
        h, w = img.shape[:2]
        s, t, ov = self.scale, self.tile, self.overlap
        out = np.empty((h * s, w * s, 3), np.uint8)

        total = math.ceil(h / t) * math.ceil(w / t)
        done = 0
        for y0 in range(0, h, t):
            for x0 in range(0, w, t):
                y1, x1 = min(y0 + t, h), min(x0 + t, w)

                # Read the tile plus an overlap margin, clamped to the image.
                py0, px0 = max(0, y0 - ov), max(0, x0 - ov)
                py1, px1 = min(h, y1 + ov), min(w, x1 + ov)

                patch = img[py0:py1, px0:px1].astype(np.float32) / 255.0
                res = self._infer(patch)

                # Discard the margin, in output coordinates.
                cy0, cx0 = (y0 - py0) * s, (x0 - px0) * s
                cy1, cx1 = cy0 + (y1 - y0) * s, cx0 + (x1 - x0) * s

                out[y0 * s : y1 * s, x0 * s : x1 * s] = np.clip(
                    res[cy0:cy1, cx0:cx1] * 255.0 + 0.5, 0, 255
                ).astype(np.uint8)

                done += 1
                if on_tile:
                    on_tile(done, total)
        return out


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


@dataclass
class Plan:
    src_size: tuple[int, int]
    final_size: tuple[int, int]
    crop_box: tuple[int, int, int, int] | None  # applied to the upscaled image
    passes: int
    ai_size: tuple[int, int]  # size after the AI passes

    @property
    def scale_factor(self) -> float:
        return self.final_size[0] / self.src_size[0]


def build_plan(
    src: tuple[int, int],
    target_long: int | None,
    exact: tuple[int, int] | None,
    fit: str,
    model_scale: int,
    max_passes: int,
    allow_downscale: bool,
) -> Plan:
    w, h = src

    if exact:
        fw, fh = exact
        # `cover` fills the canvas and crops the excess; `contain` fits inside.
        pick = max if fit == "cover" else min
        needed = pick(fw / w, fh / h)
    else:
        assert target_long is not None
        needed = target_long / max(w, h)
        fw, fh = None, None  # type: ignore[assignment]

    # An image already at or above the target is left alone unless the user
    # explicitly opts in to shrinking it -- never silently discard resolution.
    clamped = False
    if needed <= 1.0 and not allow_downscale:
        needed, clamped = 1.0, True

    # How many 4x passes to reach (or exceed) the needed factor.
    passes, acc = 0, 1.0
    while acc < needed - 1e-9 and passes < max_passes:
        acc *= model_scale
        passes += 1

    ai_w, ai_h = w * int(acc), h * int(acc)

    if exact:
        if fit == "cover":
            # Scale to cover, then centre-crop to the exact canvas.
            k = max(fw / ai_w, fh / ai_h)
            rw, rh = max(fw, round(ai_w * k)), max(fh, round(ai_h * k))
            left, top = (rw - fw) // 2, (rh - fh) // 2
            return Plan(src, (fw, fh), (left, top, left + fw, top + fh), passes, (rw, rh))
        # contain: preserve aspect ratio, no crop, may be smaller than canvas
        k = min(fw / ai_w, fh / ai_h)
        return Plan(src, (max(1, round(ai_w * k)), max(1, round(ai_h * k))), None, passes, (0, 0))

    if clamped:
        # Already big enough: pass it through at its original size.
        return Plan(src, (w, h), None, 0, (w, h))

    # Nail the long edge exactly, so rounding never drifts off the target.
    if w >= h:
        final = (target_long, max(1, round(h * target_long / w)))
    else:
        final = (max(1, round(w * target_long / h)), target_long)
    return Plan(src, final, None, passes, (ai_w, ai_h))


# --------------------------------------------------------------------------- #
# Per-image processing
# --------------------------------------------------------------------------- #


def load_image(path: Path) -> tuple[Image.Image, Image.Image | None, bytes | None]:
    """Return (rgb, alpha_or_None, icc_profile_or_None), EXIF rotation applied."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # honour camera orientation
    icc = img.info.get("icc_profile")

    has_alpha = img.mode in ("RGBA", "LA", "PA") or (
        img.mode == "P" and "transparency" in img.info
    )
    if has_alpha:
        rgba = img.convert("RGBA")
        return rgba.convert("RGB"), rgba.getchannel("A"), icc
    return img.convert("RGB"), None, icc


def save_image(
    img: Image.Image,
    path: Path,
    fmt: str,
    quality: int,
    icc: bytes | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params: dict = {}
    if icc:
        params["icc_profile"] = icc

    if fmt == "png":
        params.update(compress_level=6)
    elif fmt == "jpeg":
        if img.mode == "RGBA":
            img = img.convert("RGB")  # JPEG has no alpha
        params.update(quality=quality, subsampling=0, progressive=True, optimize=True)
    elif fmt == "webp":
        params.update(quality=quality, method=6)
    elif fmt == "tiff":
        params.update(compression="tiff_lzw")

    img.save(path, format=fmt.upper(), **params)


def process_image(
    src_path: Path,
    dst_path: Path,
    plan: Plan,
    upscaler: OnnxUpscaler | None,
    sharpen: float,
    fmt: str,
    quality: int,
    verbose: bool,
) -> None:
    rgb, alpha, icc = load_image(src_path)

    arr = np.asarray(rgb, dtype=np.uint8)
    del rgb

    started = time.time()
    for p in range(plan.passes):
        if upscaler is None:
            break

        tty = sys.stdout.isatty()
        last = [0.0]

        def on_tile(done: int, total: int, _p=p) -> None:
            if not verbose:
                return
            now = time.time()
            # On a terminal, redraw one line. When piped or logged, emit a line
            # every few seconds instead of thousands of carriage returns.
            if not tty:
                if done != total and now - last[0] < 15.0:
                    return
                last[0] = now
            elapsed = now - started
            eta = (elapsed / done) * (total - done) if done else 0.0
            msg = (f"    pass {_p + 1}/{plan.passes}  tile {done}/{total}"
                   f"  ({100 * done / total:3.0f}%)  eta {human_time(eta)}")
            print(f"\r{msg}" if tty else msg, end="" if tty else "\n", flush=True)

        arr = upscaler.upscale(arr, on_tile=on_tile)
        if verbose and tty:
            print()

    img = Image.fromarray(arr)
    del arr

    # Resize/crop down to the exact final geometry.
    if plan.crop_box is not None:
        if img.size != plan.ai_size:
            img = img.resize(plan.ai_size, Image.LANCZOS)
        img = img.crop(plan.crop_box)
    elif img.size != plan.final_size:
        img = img.resize(plan.final_size, Image.LANCZOS)

    if sharpen > 0:
        img = img.filter(ImageFilter.UnsharpMask(radius=2.0, percent=int(sharpen * 100), threshold=3))

    if alpha is not None:
        img = img.convert("RGBA")
        img.putalpha(alpha.resize(img.size, Image.LANCZOS))

    save_image(img, dst_path, fmt, quality, icc)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_exact(value: str) -> tuple[int, int]:
    try:
        w, h = value.lower().replace("×", "x").split("x")
        return int(w), int(h)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"expected WxH, e.g. 7680x4320 (got {value!r})") from exc


def collect_inputs(paths: list[Path], recursive: bool) -> list[Path]:
    found: list[Path] = []
    for p in paths:
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            found += sorted(f for f in it if f.is_file() and f.suffix.lower() in READ_EXTS)
        elif p.is_file():
            found.append(p)
        else:
            print(f"warning: {p} does not exist, skipping", file=sys.stderr)
    # de-dupe, keep order
    seen, out = set(), []
    for f in found:
        if f.resolve() not in seen:
            seen.add(f.resolve())
            out.append(f)
    return out


def human_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Upscale images to 8K with Real-ESRGAN (ONNX Runtime, CPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="models:\n"
        + "\n".join(f"  {k:<8} {v[3]}" for k, v in MODELS.items())
        + "\n\ntargets:\n"
        + "\n".join(f"  {k:<8} {v}px long edge" for k, v in TARGETS.items()),
    )
    ap.add_argument("-i", "--input", type=Path, nargs="+", default=[HERE / "input"],
                    help="input files and/or directories (default: ./input)")
    ap.add_argument("-o", "--output", type=Path, default=HERE / "output",
                    help="output directory (default: ./output)")
    ap.add_argument("-R", "--recursive", action="store_true",
                    help="recurse into input directories")

    ap.add_argument("-t", "--target", default="8k", choices=sorted(TARGETS),
                    help="named target, sets the long edge (default: 8k = 7680px)")
    ap.add_argument("--long-edge", type=int,
                    help="explicit long edge in px, overrides --target")
    ap.add_argument("--exact", type=parse_exact, metavar="WxH",
                    help="exact output canvas, e.g. 7680x4320")
    ap.add_argument("--fit", choices=["cover", "contain"], default="cover",
                    help="with --exact: crop to fill (cover) or letterbox-free fit (contain)")
    ap.add_argument("--allow-downscale", action="store_true",
                    help="also shrink images that are already larger than the target")

    ap.add_argument("-m", "--method", choices=[*MODELS, "lanczos"], default="quality",
                    help="upscaling method (default: quality)")
    ap.add_argument("--tile", type=int, default=512,
                    help="tile size for model inference (default: 512; larger is "
                         "faster and uses more RAM)")
    ap.add_argument("--overlap", type=int, default=None,
                    help="tile overlap margin, discarded from output "
                         "(default: per-model, 96 for quality / 32 for fast)")
    ap.add_argument("--threads", type=int, help="inference threads (default: all cores)")
    ap.add_argument("--max-passes", type=int, default=3,
                    help="cap on 4x model passes (default: 3, i.e. up to 64x)")

    ap.add_argument("-f", "--format", dest="fmt", default="png",
                    choices=["png", "jpeg", "webp", "tiff"], help="output format (default: png)")
    ap.add_argument("-q", "--quality", type=int, default=95,
                    help="quality for jpeg/webp (default: 95)")
    ap.add_argument("--sharpen", type=float, default=None, metavar="AMOUNT",
                    help="unsharp mask strength 0..2 (default: 0 for AI, 0.6 for lanczos)")

    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    ap.add_argument("-n", "--dry-run", action="store_true", help="show the plan and exit")
    ap.add_argument("-Q", "--quiet", action="store_true", help="less output")

    args = ap.parse_args(argv)
    verbose = not args.quiet

    files = collect_inputs(args.input, args.recursive)
    if not files:
        where = ", ".join(str(p) for p in args.input)
        print(f"No images found in: {where}", file=sys.stderr)
        print(f"Supported extensions: {' '.join(sorted(READ_EXTS))}", file=sys.stderr)
        return 1

    target_long = args.long_edge or TARGETS[args.target]
    sharpen = args.sharpen if args.sharpen is not None else (0.6 if args.method == "lanczos" else 0.0)

    # Load the model once and reuse it across the batch.
    upscaler = None
    model_scale = 4
    if args.method != "lanczos":
        fname, model_scale, min_overlap, desc = MODELS[args.method]
        overlap = args.overlap if args.overlap is not None else min_overlap
        model_path = MODEL_DIR / fname
        if not model_path.exists():
            print(f"error: model not found: {model_path}", file=sys.stderr)
            print("Run ./scripts/fetch_models.sh to download it.", file=sys.stderr)
            return 2
        if verbose:
            print(f"Model: {desc}")
            print(f"Tiling: {args.tile}px tiles, {overlap}px overlap")
        if args.overlap is not None and args.overlap < min_overlap:
            print(f"warning: --overlap {args.overlap} is below the {min_overlap}px this "
                  f"model needs; expect visible tile seams", file=sys.stderr)
        if not args.dry_run:
            try:
                upscaler = OnnxUpscaler(
                    model_path, model_scale, args.tile, overlap, args.threads
                )
            except Exception as exc:
                print(f"warning: could not load model ({exc}); falling back to Lanczos",
                      file=sys.stderr)
                upscaler, model_scale = None, 1
                sharpen = sharpen or 0.6

    if upscaler is None and not args.dry_run:
        model_scale = 1  # no AI passes possible

    ok, failed, skipped = 0, 0, 0
    batch_start = time.time()

    for idx, src in enumerate(files, 1):
        try:
            with Image.open(src) as probe:
                probe = ImageOps.exif_transpose(probe)
                src_size = probe.size
        except Exception as exc:
            print(f"[{idx}/{len(files)}] {src.name}: cannot read ({exc})", file=sys.stderr)
            failed += 1
            continue

        plan = build_plan(
            src_size, None if args.exact else target_long, args.exact, args.fit,
            model_scale if model_scale > 1 else 4,
            args.max_passes if upscaler is not None or args.dry_run else 0,
            args.allow_downscale,
        )
        if upscaler is None and not args.dry_run:
            plan.passes = 0

        dst = args.output / f"{src.stem}_{plan.final_size[0]}x{plan.final_size[1]}.{args.fmt}"

        if verbose or args.dry_run:
            print(
                f"[{idx}/{len(files)}] {src.name}: "
                f"{src_size[0]}x{src_size[1]} -> {plan.final_size[0]}x{plan.final_size[1]}"
                f"  ({plan.scale_factor:.2f}x, {plan.passes} AI pass"
                f"{'es' if plan.passes != 1 else ''})"
            )
            if plan.passes and plan.ai_size[0] * plan.ai_size[1] > INTERMEDIATE_PIXEL_WARN:
                mp = plan.ai_size[0] * plan.ai_size[1] / 1e6
                print(f"    note: {plan.ai_size[0]}x{plan.ai_size[1]} intermediate "
                      f"({mp:.0f} MP) -- this one is heavy")

        if args.dry_run:
            continue

        if dst.exists() and not args.force:
            if verbose:
                print("    exists, skipping (use --force to overwrite)")
            skipped += 1
            continue

        t0 = time.time()
        try:
            process_image(src, dst, plan, upscaler, sharpen, args.fmt, args.quality, verbose)
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed += 1
            continue

        ok += 1
        if verbose:
            size_mb = dst.stat().st_size / 1e6
            print(f"    wrote {dst.name}  {size_mb:.1f} MB  in {human_time(time.time() - t0)}")

    if args.dry_run:
        print(f"\nDry run: {len(files)} image(s) planned, nothing written.")
        return 0

    if verbose:
        print(f"\nDone: {ok} written, {skipped} skipped, {failed} failed "
              f"in {human_time(time.time() - batch_start)}")
        if ok:
            print(f"Output: {args.output}")
    return 1 if failed and not ok else 0


if __name__ == "__main__":
    sys.exit(main())
