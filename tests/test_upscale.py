"""Tests for upscale.py. Run with: .venv/bin/python -m pytest tests/ -v

The model-backed tests are skipped automatically if models/ is empty, so the
geometry and I/O tests still run in a bare checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import upscale as U

FAST_MODEL = U.MODEL_DIR / U.MODELS["fast"][0]
needs_model = pytest.mark.skipif(
    not FAST_MODEL.exists(), reason="run scripts/fetch_models.sh first"
)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "src,target,expected,passes",
    [
        ((1920, 1080), 7680, (7680, 4320), 1),   # exactly 4x -> one pass
        ((960, 640), 7680, (7680, 5120), 2),     # 8x -> two passes
        ((1080, 1920), 7680, (4320, 7680), 1),   # portrait: long edge is height
        ((3840, 2160), 7680, (7680, 4320), 1),   # 2x still needs a pass
        ((7680, 4320), 7680, (7680, 4320), 0),   # already there -> no work
        ((1000, 1000), 7680, (7680, 7680), 2),   # square
    ],
)
def test_long_edge_plans(src, target, expected, passes):
    p = U.build_plan(src, target, None, "cover", 4, 3, False)
    assert p.final_size == expected
    assert p.passes == passes


def test_aspect_ratio_preserved_on_long_edge():
    src = (1234, 987)
    p = U.build_plan(src, 7680, None, "cover", 4, 3, False)
    assert abs(p.final_size[0] / p.final_size[1] - src[0] / src[1]) < 0.002


def test_exact_cover_fills_canvas_and_crops():
    p = U.build_plan((960, 640), None, (7680, 4320), "cover", 4, 3, False)
    assert p.final_size == (7680, 4320)
    assert p.crop_box is not None
    left, top, right, bottom = p.crop_box
    assert right - left == 7680 and bottom - top == 4320


def test_exact_contain_fits_inside_canvas():
    p = U.build_plan((960, 640), None, (7680, 4320), "contain", 4, 3, False)
    assert p.final_size[0] <= 7680 and p.final_size[1] <= 4320
    assert p.crop_box is None
    assert abs(p.final_size[0] / p.final_size[1] - 960 / 640) < 0.01


def test_oversized_input_untouched_without_flag():
    p = U.build_plan((12000, 8000), 7680, None, "cover", 4, 3, False)
    assert p.passes == 0
    assert p.final_size == (12000, 8000)


def test_oversized_input_shrinks_with_flag():
    p = U.build_plan((12000, 8000), 7680, None, "cover", 4, 3, True)
    assert p.passes == 0
    assert p.final_size == (7680, 5120)


def test_max_passes_is_respected():
    p = U.build_plan((100, 100), 7680, None, "cover", 4, 1, False)
    assert p.passes == 1


def test_parse_exact():
    assert U.parse_exact("7680x4320") == (7680, 4320)
    assert U.parse_exact("7680X4320") == (7680, 4320)
    assert U.parse_exact("7680×4320") == (7680, 4320)
    with pytest.raises(Exception):
        U.parse_exact("not-a-size")


# --------------------------------------------------------------------------- #
# Model inference
# --------------------------------------------------------------------------- #


@needs_model
def test_model_scales_by_four():
    img = (np.random.rand(48, 64, 3) * 255).astype(np.uint8)
    out = U.OnnxUpscaler(FAST_MODEL, tile=256).upscale(img)
    assert out.shape == (48 * 4, 64 * 4, 3)
    assert out.dtype == np.uint8


def photo_like(w: int = 96, h: int = 96) -> np.ndarray:
    """Structured image standing in for a photo: gradients, edges, fine detail.

    Pure random noise is a pathological case for tiled inference -- it has no
    spatial correlation for the model to exploit, so it overstates edge error.
    Real photographs behave like this instead.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 128 + 100 * np.sin(xx / 9.0)
    g = (yy / max(1, h - 1)) * 255.0
    b = 90 + 80 * np.cos((xx + yy) / 13.0)
    img = np.stack([r, g, b], -1)
    img[h // 3 : 2 * h // 3, w // 4 : w // 2] = 240.0   # hard-edged block
    img[:, ::7] = 20.0                                   # fine repeating lines
    return np.clip(img, 0, 255).astype(np.uint8)


def tiling_error(model_path, img, tile, overlap):
    """Max/mean deviation of tiled inference from whole-image inference."""
    whole = U.OnnxUpscaler(model_path, tile=8192, overlap=8).upscale(img)
    tiled = U.OnnxUpscaler(model_path, tile=tile, overlap=overlap).upscale(img)
    d = np.abs(tiled.astype(np.int16) - whole.astype(np.int16))
    return d.max(), d.mean()


@needs_model
def test_shipped_defaults_are_effectively_lossless():
    """The defaults we ship must match whole-image inference.

    This is the headline correctness property: tiling is an implementation
    detail for bounding memory, and must not change the picture.
    """
    img = photo_like(600, 400)  # forces a real tile grid at tile=512
    _, _, min_overlap, _ = U.MODELS["fast"]
    max_d, mean_d = tiling_error(FAST_MODEL, img, tile=512, overlap=min_overlap)
    assert max_d <= 2, f"seams at shipped defaults, max diff {max_d}"
    assert mean_d < 0.01, f"mean drift {mean_d}"


@needs_model
def test_model_min_overlap_is_sufficient():
    """Each model's declared min_overlap must actually suppress seams."""
    img = photo_like(600, 400)
    for name, (fname, _, min_overlap, _) in U.MODELS.items():
        path = U.MODEL_DIR / fname
        if not path.exists():
            pytest.skip(f"{name} model not downloaded")
        max_d, _ = tiling_error(path, img, tile=512, overlap=min_overlap)
        assert max_d <= 2, f"{name}: min_overlap={min_overlap} leaves seams (max {max_d})"


@needs_model
def test_insufficient_overlap_is_measurably_worse():
    """Guards against anyone 'optimising' the overlap margin back down.

    Without this, shrinking min_overlap would silently reintroduce seams --
    the exact regression this suite exists to prevent.
    """
    img = photo_like(600, 400)
    good_max, _ = tiling_error(FAST_MODEL, img, tile=512, overlap=U.MODELS["fast"][2])
    bad_max, _ = tiling_error(FAST_MODEL, img, tile=512, overlap=0)
    assert bad_max > good_max * 3, f"expected clear degradation: {bad_max} vs {good_max}"


@needs_model
def test_tiling_bounded_even_on_pure_noise():
    """Worst-case content: maximum entropy, no structure to exploit."""
    rng = np.random.default_rng(0)
    img = (rng.random((600, 400, 3)) * 255).astype(np.uint8)
    max_d, mean_d = tiling_error(FAST_MODEL, img, tile=512, overlap=U.MODELS["fast"][2])
    assert max_d <= 12, f"max diff {max_d}"
    assert mean_d < 0.5, f"mean diff {mean_d}"


@needs_model
def test_non_multiple_dimensions_survive_tiling():
    """Odd sizes must not lose or duplicate edge rows/columns."""
    img = (np.random.rand(70, 37, 3) * 255).astype(np.uint8)
    out = U.OnnxUpscaler(FAST_MODEL, tile=32, overlap=16).upscale(img)
    assert out.shape == (280, 148, 3)


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #


def _plan_to(src_size, long_edge):
    return U.build_plan(src_size, long_edge, None, "cover", 4, 3, False)


def test_lanczos_path_needs_no_model(tmp_path):
    src = tmp_path / "in.png"
    Image.new("RGB", (100, 50), (30, 120, 200)).save(src)
    plan = U.Plan((100, 50), (400, 200), None, 0, (0, 0))
    dst = tmp_path / "out.png"
    U.process_image(src, dst, plan, None, 0.6, "png", 95, verbose=False)
    assert Image.open(dst).size == (400, 200)


def test_alpha_channel_is_preserved(tmp_path):
    src = tmp_path / "in.png"
    im = Image.new("RGBA", (40, 40), (255, 0, 0, 255))
    im.putalpha(Image.linear_gradient("L").resize((40, 40)))
    im.save(src)

    dst = tmp_path / "out.png"
    plan = U.Plan((40, 40), (160, 160), None, 0, (0, 0))
    U.process_image(src, dst, plan, None, 0.0, "png", 95, verbose=False)

    out = Image.open(dst)
    assert out.mode == "RGBA"
    assert out.size == (160, 160)
    a = np.asarray(out.getchannel("A"))
    assert a.min() < 40 and a.max() > 215, "alpha gradient lost"


def test_exif_orientation_is_applied(tmp_path):
    """A portrait photo tagged 'rotate 90' must come out portrait."""
    src = tmp_path / "rot.jpg"
    im = Image.new("RGB", (80, 40), (10, 200, 10))
    exif = im.getexif()
    exif[274] = 6  # Orientation = rotate 90 CW
    im.save(src, exif=exif)

    with Image.open(src) as probe:
        from PIL import ImageOps
        assert ImageOps.exif_transpose(probe).size == (40, 80)

    dst = tmp_path / "out.png"
    plan = U.Plan((40, 80), (80, 160), None, 0, (0, 0))
    U.process_image(src, dst, plan, None, 0.0, "png", 95, verbose=False)
    assert Image.open(dst).size == (80, 160)


@pytest.mark.parametrize("fmt,mode", [("png", "RGB"), ("jpeg", "RGB"), ("webp", None), ("tiff", "RGB")])
def test_output_formats(tmp_path, fmt, mode):
    src = tmp_path / "in.png"
    Image.new("RGB", (60, 40), (90, 90, 200)).save(src)
    dst = tmp_path / f"out.{fmt}"
    plan = U.Plan((60, 40), (240, 160), None, 0, (0, 0))
    U.process_image(src, dst, plan, None, 0.0, fmt, 92, verbose=False)
    out = Image.open(dst)
    assert out.size == (240, 160)
    if mode:
        assert out.mode == mode


def test_jpeg_drops_alpha_instead_of_crashing(tmp_path):
    src = tmp_path / "in.png"
    Image.new("RGBA", (40, 40), (255, 0, 0, 128)).save(src)
    dst = tmp_path / "out.jpeg"
    plan = U.Plan((40, 40), (160, 160), None, 0, (0, 0))
    U.process_image(src, dst, plan, None, 0.0, "jpeg", 92, verbose=False)
    assert Image.open(dst).size == (160, 160)


def test_exact_cover_output_is_exact(tmp_path):
    src = tmp_path / "in.png"
    Image.new("RGB", (100, 100), (200, 50, 50)).save(src)  # square -> 16:9 needs crop
    plan = U.build_plan((100, 100), None, (400, 225), "cover", 4, 3, False)
    dst = tmp_path / "out.png"
    U.process_image(src, dst, plan, None, 0.0, "png", 95, verbose=False)
    assert Image.open(dst).size == (400, 225)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_reports_no_images(tmp_path, capsys):
    assert U.main(["-i", str(tmp_path), "-o", str(tmp_path / "o")]) == 1
    assert "No images found" in capsys.readouterr().err


def test_cli_dry_run_writes_nothing(tmp_path, capsys):
    inp = tmp_path / "in"
    inp.mkdir()
    Image.new("RGB", (200, 100), (1, 2, 3)).save(inp / "a.png")
    out = tmp_path / "out"
    assert U.main(["-i", str(inp), "-o", str(out), "--dry-run", "-m", "lanczos"]) == 0
    assert "Dry run" in capsys.readouterr().out
    assert not out.exists() or not list(out.iterdir())


def test_cli_lanczos_end_to_end(tmp_path):
    inp = tmp_path / "in"
    inp.mkdir()
    Image.new("RGB", (320, 180), (70, 130, 180)).save(inp / "a.png")
    out = tmp_path / "out"
    assert U.main(["-i", str(inp), "-o", str(out), "-m", "lanczos", "-t", "4k", "-Q"]) == 0
    produced = list(out.glob("*.png"))
    assert len(produced) == 1
    assert Image.open(produced[0]).size == (3840, 2160)
    assert "3840x2160" in produced[0].name


def test_cli_skips_existing_then_forces(tmp_path):
    inp = tmp_path / "in"
    inp.mkdir()
    Image.new("RGB", (100, 100), (5, 5, 5)).save(inp / "a.png")
    out = tmp_path / "out"
    args = ["-i", str(inp), "-o", str(out), "-m", "lanczos", "--long-edge", "200", "-Q"]
    assert U.main(args) == 0
    target = out / "a_200x200.png"
    assert target.exists()

    target.write_bytes(b"")  # corrupt it; a skip must leave it alone
    assert U.main(args) == 0
    assert target.stat().st_size == 0

    assert U.main([*args, "--force"]) == 0
    assert target.stat().st_size > 0


def test_cli_recursive_discovery(tmp_path):
    inp = tmp_path / "in"
    (inp / "nested").mkdir(parents=True)
    Image.new("RGB", (50, 50), (1, 1, 1)).save(inp / "top.png")
    Image.new("RGB", (50, 50), (2, 2, 2)).save(inp / "nested" / "deep.png")
    out = tmp_path / "out"
    args = ["-i", str(inp), "-o", str(out), "-m", "lanczos", "--long-edge", "100", "-Q"]

    U.main(args)
    assert len(list(out.glob("*.png"))) == 1  # non-recursive

    U.main([*args, "-R", "--force"])
    assert len(list(out.glob("*.png"))) == 2


def test_cli_continues_past_corrupt_file(tmp_path, capsys):
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "broken.png").write_bytes(b"definitely not a png")
    Image.new("RGB", (50, 50), (9, 9, 9)).save(inp / "good.png")
    out = tmp_path / "out"
    rc = U.main(["-i", str(inp), "-o", str(out), "-m", "lanczos", "--long-edge", "100", "-Q"])
    assert rc == 0  # one failure but one success
    assert len(list(out.glob("*.png"))) == 1
    assert "cannot read" in capsys.readouterr().err


def test_collect_inputs_ignores_non_images(tmp_path):
    (tmp_path / "a.png").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    (tmp_path / "b.JPG").write_bytes(b"")
    found = {p.name for p in U.collect_inputs([tmp_path], False)}
    assert found == {"a.png", "b.JPG"}


def test_collect_inputs_dedupes(tmp_path):
    f = tmp_path / "a.png"
    f.write_bytes(b"")
    assert len(U.collect_inputs([f, f, tmp_path], False)) == 1
