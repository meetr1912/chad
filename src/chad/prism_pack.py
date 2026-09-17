"""Loader for Prism ML's Hadamard-folded ternary packs (`prism_hadamard_qwen35`).

These packs are a Qwen3.8-27B checkpoint whose projections are stored in a rotated
basis: each weight matrix is multiplied by a fixed sign vector and transformed by a
blockwise Hadamard rotation offline, then quantized to 2-bit affine g128 whose three
levels reproduce the ternary set {-s, 0, +s}. The rotation costs no extra bits and no
extra weight traffic, but it is not optional: the matching transform has to be applied
to the ACTIVATIONS at runtime, and the embedding table needs the inverse transform
after its lookup. mlx-lm's ordinary affine loader would find weights/scales/biases of
exactly the right shapes, skip both transforms, and return plausible-looking garbage
rather than an error — so the pack is routed here on `model_type` instead.

The pack ships its own `runtime/` Python and asks callers to `sys.path.insert` it.
chad does not import code out of a model download; `Packed` below is a reimplementation
of that module's forward, and `load()` refuses any pack whose declared quantization is
not the 2-bit affine g128 container it was written against.
"""

import json
import math
import os
from typing import Any, Optional, cast

from .diag import log

_MODEL_TYPE = "prism_hadamard_qwen35"
_BLOCKS = (512, 1024, 2048, 4096)


def is_prism_pack(config: dict) -> bool:
    return config.get("model_type") == _MODEL_TYPE


def _hadamard(x, block: int, signs, inverse: bool):
    """Apply the pack's blockwise Hadamard rotation to activations.

    fp32 throughout: the transform sums `block` terms, and at fp16 the accumulation
    loses the low bits the ternary levels are meant to resolve."""
    import mlx.core as mx

    shape, dtype = x.shape, x.dtype
    if shape[-1] % block:
        raise ValueError(f"Hadamard block {block} does not divide width {shape[-1]}")
    x = x.astype(mx.float32)
    if not inverse:
        x = x * signs
    x = mx.hadamard_transform(x.reshape(-1, block),
                              scale=1 / math.sqrt(block)).reshape(shape)
    if inverse:
        x = x * signs
    return x.astype(dtype)


def _packed_class():
    import mlx.core as mx
    import mlx.nn as nn

    class Packed(nn.Module):
        """One rotated projection (or the rotated embedding table).

        Stands in for `nn.QuantizedLinear` / `nn.Embedding`. `block == 0` means the
        tensor was stored unrotated and this is a plain 2-bit affine matmul."""

        def __init__(self, weight, scales, biases, block, signs, embedding, dtype):
            super().__init__()
            self.weight, self.scales, self.biases = weight, scales, biases
            if signs is not None:
                self.signs = signs
            self.block, self.embedding, self.dtype = block, embedding, dtype
            # chad's fused-projection paths read these off QuantizedLinear.
            self.bits, self.group_size = 2, 128

        def __call__(self, x):
            if self.embedding:
                shape = x.shape
                out = mx.dequantize(
                    self.weight[x.reshape(-1)], self.scales[x.reshape(-1)],
                    self.biases[x.reshape(-1)], group_size=128, bits=2,
                ).reshape(*shape, -1).astype(self.dtype)
                return (_hadamard(out, self.block, self.signs, True)
                        if self.block else out)
            if self.block:
                x = _hadamard(x, self.block, self.signs, False)
            return mx.quantized_matmul(x, self.weight, self.scales, self.biases,
                                       transpose=True, group_size=128, bits=2)

    return Packed


def load(model_path: str, config: Optional[dict] = None) -> tuple[Any, dict]:
    """Build the qwen3_5 model these packs carry, with rotated projections in place.

    Returns what `mlx_lm.utils.load_model` returns: (model, config). The pack bundles
    an FP16 vision tower chad has no use for; only the `language_model.` half is read,
    which is also 0.92 GB less resident memory."""
    import mlx.core as mx
    from mlx_lm.models import qwen3_5 as q35

    if config is None:
        with open(os.path.join(model_path, "config.json")) as f:
            config = json.load(f)
    if not is_prism_pack(config):
        raise ValueError(f"not a Prism pack: model_type={config.get('model_type')!r}")
    quant = config.get("quantization") or {}
    if (quant.get("bits"), quant.get("group_size"), quant.get("mode")) != (2, 128, "affine"):
        raise ValueError(f"unsupported Prism container {quant!r}; this loader "
                         "implements 2-bit affine g128 only")

    model = q35.Model(q35.ModelArgs(model_type="qwen3_5",
                                    text_config=config["text_config"]))
    # SAFETY: mx.load's return type widens across .npy/.npz, but it returns a
    # name -> array mapping for every .safetensors input, which is the only
    # extension this loader ever passes it.
    loaded = cast(dict, mx.load(os.path.join(model_path, "model.safetensors")))
    weights = {k: v for k, v in loaded.items() if k.startswith("language_model.")}
    del loaded
    Packed = _packed_class()
    seen: set[str] = set()
    for record in config["modules"]:
        path = record["path"]
        if path in seen:
            raise ValueError(f"duplicate packed module {path}")
        seen.add(path)
        block = record["block"]
        if block and block not in _BLOCKS:
            raise ValueError(f"unsupported Hadamard block {block}")
        key = "language_model." + path
        signs = weights.get(key + ".signs")
        if block and signs is None:
            raise ValueError(f"{path}: rotated module with no sign vector")
        # mlx Modules are dicts, so the pack's dotted paths walk by subscript; a
        # numeric segment indexes the plain list `model.layers` is.
        parts = key.split(".")
        parent = model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else parent[part]
        parent[parts[-1]] = Packed(
            weights[key + ".weight"], weights[key + ".scales"],
            weights[key + ".biases"], block, signs, record["embedding"], mx.float16)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    log.info("Prism pack loaded: %d rotated modules, 2-bit affine g128, "
             "vision tower skipped", len(seen))
    return model, config
