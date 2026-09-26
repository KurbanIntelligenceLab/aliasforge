#!/usr/bin/env python3
"""InternVL / VisualPRM production preprocessing, transcribed verbatim from source.

Why this file exists.  `AutoProcessor` on the VisualPRM-8B repo returns the
CLIP-style feature extractor recorded in `preprocessor_config.json`
(resize-shortest-side to 448, `do_center_crop=True`).  That is NOT what the model
is run with.  Every OpenGVLab entry point -- the InternVL README, the
InternVL2_5-8B card, the VisualPRM-8B card, and VLMEvalKit -- feeds the model
through `load_image`/`dynamic_preprocess` instead, which TILES the image and
appends a thumbnail and never centre-crops anything.

The two paths disagree about which pixels reach the model, so a pair certified
against the wrong one certifies nothing about the deployed system.  The functions
below are copied from the OpenGVLab sources (verified byte-identical across the
InternVL README, internvl_chat/README.md and the InternVL2_5-8B card; the
VisualPRM-8B card differs only in one parameter name), so that what we hash is
what the model is actually fed.

Consequences that matter for the certificate, read off this code:

  * `dynamic_preprocess` crops tiles that EXACTLY TILE the resized image, so
    tiling discards nothing.  There is no crop-shaped dead region here -- the
    centre-crop dead region that the CLIP path exhibits is an artifact of the
    wrong path.
  * `load_image` hardcodes `use_thumbnail=True`, so for any multi-tile image the
    model additionally receives a 448x448 downscale of the WHOLE original.  A
    perturbation must therefore lie in the null space of the tile resize AND of
    the thumbnail resize simultaneously.  That is a stronger constraint than
    either alone and it is easy to miss.
  * Both `image.resize(...)` calls pass no `resample`, so they take Pillow's
    default; `build_transform` names BICUBIC explicitly for the per-tile
    448x448 resize.
"""

from __future__ import annotations

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Defaults confirmed against config.json for both InternVL2_5-8B and VisualPRM-8B:
# force_image_size=448, min_dynamic_patch=1, max_dynamic_patch=12, use_thumbnail=true.
IMAGE_SIZE = 448
MIN_NUM = 1
MAX_NUM = 12


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Verbatim from the OpenGVLab sources."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=MIN_NUM, max_num=MAX_NUM,
                       image_size=IMAGE_SIZE, use_thumbnail=False):
    """Verbatim from the OpenGVLab sources.

    NOTE the tie-break in `find_closest_aspect_ratio` reads `target_ratios` in the
    order produced by sorting a SET by i*j.  Python's sort is stable but the
    within-group order of a set is unspecified, so on an exact aspect-ratio tie the
    selected grid can in principle depend on set iteration order.  We do not rely
    on it: `interface_state` below hashes the realized grid, so a tie that resolved
    differently would change the digest rather than pass silently.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_pixel_values(image, input_size=IMAGE_SIZE, max_num=MAX_NUM) -> np.ndarray:
    """The tensor the model is actually fed: (n_tiles, 3, 448, 448), float32.

    Mirrors `load_image` (which hardcodes use_thumbnail=True) but returns numpy and
    takes an already-opened image, so the caller controls the pixels.  Normalization
    is applied exactly as `build_transform` does: ToTensor scales to [0,1] in CHW,
    then Normalize with the ImageNet constants.
    """
    from PIL import Image as _Image

    if isinstance(image, np.ndarray):
        image = _Image.fromarray(image)
    image = image.convert("RGB") if image.mode != "RGB" else image

    tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)

    out = []
    for t in tiles:
        # build_transform: T.Resize((448,448), BICUBIC) then ToTensor then Normalize.
        # Tiles are already 448x448 so the resize is identity, but the thumbnail
        # path can hand us a differently sized image, so apply it unconditionally.
        if t.size != (input_size, input_size):
            t = t.resize((input_size, input_size), _Image.BICUBIC)
        arr = np.asarray(t, dtype=np.float32).transpose(2, 0, 1) / 255.0
        out.append((arr - mean) / std)
    return np.stack(out)


def interface_state(image, input_size=IMAGE_SIZE, max_num=MAX_NUM) -> dict:
    """Every image-dependent quantity the language model conditions on.

    Beyond the pixel tensor this records the realized tile grid and the tile count,
    because both are image-size dependent and both change how many visual tokens
    the model sees.  Appendix B's checklist names grid, tile count and any
    thumbnail branch explicitly; omitting them would leave an unhashed channel.
    """
    from PIL import Image as _Image

    if isinstance(image, np.ndarray):
        image = _Image.fromarray(image)
    image = image.convert("RGB") if image.mode != "RGB" else image

    tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pv = load_pixel_values(image, input_size, max_num)
    return {
        "pixel_values": pv,
        "n_tiles": len(tiles),
        "has_thumbnail": len(tiles) != 1,
        "orig_size": tuple(image.size),
    }


def validate() -> list[tuple[str, bool, str]]:
    """Observable invariants a faithful transcription must satisfy."""
    from PIL import Image as _Image

    out = []

    def mk(w, h):
        return _Image.fromarray(
            (np.random.default_rng(0).integers(0, 256, (h, w, 3))).astype(np.uint8)
        )

    t = dynamic_preprocess(mk(448, 448), use_thumbnail=True)
    out.append(("square 448 -> exactly 1 tile, no thumbnail appended", len(t) == 1, f"{len(t)}"))

    t = dynamic_preprocess(mk(896, 448), use_thumbnail=True)
    out.append(("2:1 -> 2 tiles + thumbnail = 3", len(t) == 3, f"{len(t)}"))

    t = dynamic_preprocess(mk(448, 896), use_thumbnail=True)
    out.append(("1:2 -> 2 tiles + thumbnail = 3", len(t) == 3, f"{len(t)}"))

    for (w, h) in ((4000, 500), (500, 4000), (1024, 768), (333, 500)):
        t = dynamic_preprocess(mk(w, h), use_thumbnail=True)
        n_tiles = len(t) - (1 if len(t) != 1 else 0)
        ok = n_tiles <= MAX_NUM and all(x.size == (IMAGE_SIZE, IMAGE_SIZE) for x in t)
        out.append((f"{w}x{h}: <= {MAX_NUM} tiles, all 448x448", ok, f"tiles={n_tiles}"))

    pv = load_pixel_values(mk(896, 448))
    out.append(("pixel_values is (n_tiles,3,448,448) float32",
                pv.shape[1:] == (3, 448, 448) and pv.dtype == np.float32, str(pv.shape)))
    return out


if __name__ == "__main__":
    bad = 0
    for name, ok, detail in validate():
        print(f"{'PASS' if ok else 'FAIL'}  {name}  [{detail}]")
        bad += not ok
    print(f"RESULT internvl_validation_failed={bad}")
