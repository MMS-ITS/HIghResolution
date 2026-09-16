# HIghResolution

Batch-upscale photographs to 8K using Real-ESRGAN, running on CPU via ONNX Runtime.
No GPU, no PyTorch, no cloud service.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./scripts/fetch_models.sh          # ~69MB of model weights
cp /your/photos/*.jpg input/
.venv/bin/python upscale.py        # -> output/name_7680x5120.png
```

## How it works

Upscaling happens in 4x AI steps until the image is at least the target size,
then a single Lanczos step brings it down to the exact target:

```
960x640  --AI 4x-->  3840x2560  --AI 4x-->  15360x10240  --Lanczos-->  7680x5120
```

Overshooting and stepping down is what makes the result crisp. Resizing straight
to the target leaves it soft, because the model only knows how to do 4x.

An image already at or above the target is **passed through untouched** — it is
never silently downscaled. Use `--allow-downscale` if you do want shrinking.

## Choosing a model

| `--method`  | Model | Speed | Use for |
|-------------|-------|-------|---------|
| `quality`   | Real-ESRGAN x4plus (RRDBNet-23) | slowest | photographs, final output — **default** |
| `fast`      | Real-ESRGAN general x4v3 | ~15x faster | previews, large batches, triage |
| `lanczos`   | none (plain resample) | instant | when you want no AI interpretation at all |

`lanczos` needs no model download and is a useful baseline: it will not invent
detail, so it stays faithful but looks soft at 8K.

## Common usage

```bash
# Default: everything in ./input -> 8K PNG in ./output
.venv/bin/python upscale.py

# Preview the plan and per-image sizes without doing the work
.venv/bin/python upscale.py --dry-run

# Fast pass over a big folder, recursively, as JPEG
.venv/bin/python upscale.py -i ~/photos -R -m fast -f jpeg -q 92

# Exact 16:9 8K canvas, centre-cropping anything that does not fit
.venv/bin/python upscale.py --exact 7680x4320 --fit cover

# Cinema 8K (8192px) instead of UHD-2 (7680px)
.venv/bin/python upscale.py --target 8k-dci
```

Named targets set the **long edge**, so aspect ratio is always preserved:
`8k` 7680 · `8k-dci` 8192 · `6k` 6144 · `5k` 5120 · `4k` 3840 · `4k-dci` 4096.
Use `--long-edge N` for anything else, or `--exact WxH` for a fixed canvas
(`--fit cover` crops to fill, `--fit contain` fits inside without cropping).

## Runtime

Measured on 8 CPU cores, 960x640 source to 8K (two AI passes):

| Method | Time |
|--------|------|
| `fast` | ~1.5 min |
| `quality` | ~20 min |

Cost scales with **output** pixels, so it is roughly constant per target size
regardless of how small the source is. A source that needs only one pass (e.g.
1920x1080 to 8K) is about 4x cheaper than one needing two. `--dry-run` reports
the pass count per image before you commit to a long batch.

Peak RAM is a few GB: intermediates are held as `uint8`, and inference is tiled.

## Tiling, and why the overlap matters

Inference runs on tiles to bound memory. Each tile is read with an overlap
margin that is then discarded, so every output pixel comes from a tile where it
sat well away from the edge. This is what avoids seams — no blending is used.

The required margin is a property of the model's receptive field, not a tuning
knob. Deviation from whole-image inference, measured on a photo (max error out
of 255):

| overlap | `quality` | `fast` |
|---------|-----------|--------|
| 0  | 192 | 110 |
| 16 | 92  | 16  |
| 32 | 43  | **1** |
| 64 | 9   | 0 |
| 96 | **1** | 0 |

`quality` needs far more margin than `fast` because RRDBNet-23 aggregates over
a much wider area. Hence the defaults: **96px for `quality`, 32px for `fast`**.
Lowering them reintroduces visible seams, and `tests/test_upscale.py` fails if
anyone tries.

Larger tiles are both faster and more accurate, since the fixed overlap cost is
amortised over more pixels. At tile=512/overlap=96 the `quality` model was
*faster* than tile=256/overlap=96 (27s vs 49s) for the same result. Default tile
is 512; drop it with `--tile 256` only if memory is tight.

## Output

PNG by default (lossless). `-f jpeg|webp|tiff` with `-q` for lossy formats.
Files are named `<stem>_<W>x<H>.<ext>`. Existing outputs are skipped unless
`--force`. EXIF orientation is applied, ICC colour profiles are carried across,
and alpha channels are preserved (resampled with Lanczos, since the models are
RGB-only).

## Tests

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests/ -v
```

36 tests covering target geometry, seam-free tiling, alpha, EXIF orientation,
output formats, and CLI behaviour. Model-dependent tests skip automatically if
`models/` is empty.

## Sharpness vs pixel fidelity

Worth knowing before you pick a method. Shrinking a real photo 4x, upscaling it
back, and comparing against the untouched original:

| method | PSNR vs original | sharpness |
|--------|------------------|-----------|
| bicubic | 27.50 dB | 2.47 |
| lanczos | **27.60 dB** | 2.66 |
| AI `fast` | 26.38 dB | 3.24 |
| AI `quality` | 25.45 dB | **5.35** |
| _(true original)_ | — | _8.72_ |

Lanczos wins on PSNR while looking clearly the worst, and that is not a bug.
Real-ESRGAN is GAN-based: it synthesises texture that is *statistically* right
but not pixel-aligned, which PSNR penalises heavily. Sharpness tells the other
half of the story — `quality` lands far closer to the real original than any
resampler, which is what the eye actually responds to.

Practical reading: use `quality` when the image is for looking at, and `lanczos`
only if you need every output pixel to be a defensible function of an input
pixel (measurement, forensics, diffing).

## Notes and limits

- **The source sets the ceiling.** 4K to 8K looks excellent; a 640x480 thumbnail
  pushed to 8K will look synthetic no matter which model runs. AI upscaling
  reconstructs *plausible* detail, it does not recover real detail that was
  never captured.
- Real-ESRGAN is trained on natural images. Faces, text, and line art are its
  known weak spots — `quality` handles them better than `fast`, but neither is a
  dedicated face or text restorer.
- Model weights are gitignored; `scripts/fetch_models.sh` fetches them.

## Credits

Models are ONNX exports of [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
by Xintao Wang et al. (BSD-3-Clause), fetched from
[anakhiu/realesrgan-onnx](https://huggingface.co/anakhiu/realesrgan-onnx) and
[Heliosoph/realesrgan-onnx](https://huggingface.co/Heliosoph/realesrgan-onnx).
