"""Loader for Prism ML's Hadamard-rotated ternary packs (`prism_hadamard_qwen35`).

Every projection in these packs is stored in a Hadamard-rotated basis: the rows were
multiplied by a fixed sign vector and put through a blockwise Walsh-Hadamard transform
offline, then quantized to affine 2-bit/group-128. The rotation costs no extra bits and
no extra weight traffic, but the matching transform has to be applied to the ACTIVATIONS
at runtime and inverted after the embedding lookup.

This matters because the failure mode is silent: the packed tensors have exactly the
shapes an ordinary MLX affine loader expects, so `mlx_lm.load` — once it knows the
model type at all — would skip both transforms and emit fluent garbage without raising.

The math (`fwht`, `Packed`) is the vendor's own, from the `runtime/` directory shipped
inside `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`, kept byte-for-byte in behaviour. What
this module adds is the namespace: the vendor pack targets `mlx_lm`'s bare `TextModel`,
while the repacks chad consumes are written in the `mlx-vlm-qwen3_5` tensor namespace
(`language_model.` prefix) against the `qwen3_5` `Model` wrapper, and declare
`schema_version: 2`.

Not installed for anything else: `load()` is called only when config.json says
`model_type: prism_hadamard_qwen35`.
"""

import json
import math
from pathlib import Path
from typing import Any, Optional, cast

import mlx.core as mx
import mlx.nn as nn

MODEL_TYPE = "prism_hadamard_qwen35"
_BLOCKS = (512, 1024, 2048, 4096)
# The pack's own dtype. bfloat16 was measured here for the sake of mlx_qmm_mma's MMA
# kernel, which takes bfloat16 operands: the quantized matmuls are indifferent to the
# choice, but the rest of the step is not, and the model decoded at 10.8 tok/s against
# float16's 24.4. The MMA path casts at its own call site instead.
_COMPUTE_DTYPE = mx.float16


def is_prism_pack(config: dict) -> bool:
    return config.get("model_type") == MODEL_TYPE


def fwht(x, block, signs, inverse=False):
    """Blockwise normalized Walsh-Hadamard transform over the last axis.

    Forward folds the sign vector in BEFORE the transform, inverse applies it AFTER —
    the two orders are not interchangeable and only one of them reconstructs the
    weights. Runs in float32 because the transform sums `block` terms.
    """
    shape, dtype = x.shape, x.dtype
    if shape[-1] % block:
        raise ValueError("Hadamard block does not divide activation width")
    x = x.astype(mx.float32)
    if not inverse:
        x = x * signs
    x = mx.hadamard_transform(x.reshape(-1, block), scale=1 / math.sqrt(block)).reshape(
        shape
    )
    if inverse:
        x = x * signs
    return x.astype(dtype)


class Packed(nn.Module):
    """A rotated affine 2-bit projection (or embedding), transform included."""

    group_size = 128
    bits = 2

    def __init__(self, arrays, block=0, signs=None, embedding=False, dtype=mx.float16):
        super().__init__()
        self.weight, self.scales, self.biases = [mx.array(a) for a in arrays]
        if self.scales.dtype != dtype:
            self.scales = self.scales.astype(dtype)
            self.biases = self.biases.astype(dtype)
        if signs is not None:
            self.signs = mx.array(signs)
        self.block, self.embedding, self.dtype = block, embedding, dtype

    def __call__(self, x):
        signs = getattr(self, "signs", None)
        if self.embedding:
            shape = x.shape
            indices = x.reshape(-1)
            out = (
                mx.dequantize(
                    self.weight[indices],
                    self.scales[indices],
                    self.biases[indices],
                    group_size=self.group_size,
                    bits=self.bits,
                )
                .reshape(*shape, -1)
                .astype(self.dtype)
            )
            return fwht(out, self.block, signs, inverse=True) if self.block else out
        if self.block:
            x = fwht(x, self.block, signs)
        # mlx_qmm_mma.qmm is mx.quantized_matmul plus the small-M MMA kernel on the
        # shapes and widths where it was measured to win — which is the speculative
        # verify band, where the stock kernel re-reads the weights once per row.
        from . import mlx_qmm_mma
        return mlx_qmm_mma.qmm(
            x, self.weight, self.scales, self.biases, self.group_size, self.bits
        )


def _validate(original, record, arrays, signs) -> None:
    """The vendor's `validate_record`, which is the only thing standing between a
    subtly mis-shaped pack and fluent garbage."""
    if not isinstance(original, (nn.Linear, nn.Embedding)):
        raise ValueError(f"Unsupported packed module target: {type(original).__name__}")
    if record["embedding"] != isinstance(original, nn.Embedding):
        raise ValueError("Packed module kind mismatch")
    rows, width = original.weight.shape
    if width % 128:
        raise ValueError("Invalid packed width")
    expected = [(rows, width // 16), (rows, width // 128), (rows, width // 128)]
    if [a.shape for a in arrays] != expected or arrays[0].dtype != mx.uint32:
        raise ValueError("Invalid packed tensor shapes or storage dtype")
    for array in arrays[1:]:
        if array.dtype not in (mx.float16, mx.float32, mx.bfloat16):
            raise ValueError("Invalid affine dtype")
        if not mx.all(mx.isfinite(array)).item():
            raise ValueError("Non-finite affine parameters")
    block = record["block"]
    if block:
        if width % block or signs is None or signs.shape != (width,):
            raise ValueError("Invalid transform dimensions")
        if not mx.all((signs == 1) | (signs == -1)).item():
            raise ValueError("Invalid sign values")
    elif signs is not None:
        raise ValueError("Unexpected sign vector")


# Parameters that stay float32 after the compute-dtype cast. The recurrence's decay
# terms are float32 by the checkpoint's own `mamba_ssm_dtype`, and the sign vectors are
# consumed inside fwht's float32 region, where a float16 copy would only be upcast again.
_KEEP_F32 = ("A_log", "dt_bias", "signs")


def _cast_compute_dtype(model: nn.Module, dtype=_COMPUTE_DTYPE) -> int:
    """Put the unquantized parameters in the pack's compute dtype.

    The pack stores its norms, convolutions and the two small GDN input projections as
    float32 while its affine parameters are float16. Left alone, the first rms_norm
    promotes the activation to float32 and every op downstream of it — including the
    quantized matmuls — runs wide for the rest of the model. Costs 34% of a decode step
    and 2.6x of an S=8 verify forward, which is most of what makes speculative decoding
    not pay on this pack.
    """
    from mlx.utils import tree_flatten, tree_unflatten

    # tree_flatten's stub widens to str; a parameter tree always flattens to pairs.
    flat = cast(list[tuple[str, mx.array]], tree_flatten(model.parameters()))
    updates = [
        (k, v.astype(dtype))
        for k, v in flat
        if v.dtype == mx.float32 and k.split(".")[-1] not in _KEEP_F32
    ]
    if updates:
        model.update(tree_unflatten(updates))
        mx.eval(model.parameters())
    return len(updates)


def load(
    model_path: str,
    config: Optional[dict] = None,
    rotate: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build a `qwen3_5` model whose packed modules carry their own transforms.

    `rotate=False` loads the same weights with every transform skipped. That is not a
    supported way to run the model — it is the A/B control that makes the rotation's
    effect measurable, since a pack loaded without it still produces confident tokens.
    """
    from mlx_lm.models import qwen3_5 as q35

    path = Path(model_path)
    config = config or json.loads((path / "config.json").read_text())
    if not is_prism_pack(config):
        raise ValueError(f"Not a Prism pack: model_type={config.get('model_type')!r}")
    if config.get("schema_version") not in (1, 2):
        raise ValueError(f"Unsupported pack schema: {config.get('schema_version')!r}")
    quant = config.get("quantization") or {}
    if (quant.get("bits"), quant.get("group_size"), quant.get("mode")) != (2, 128, "affine"):
        raise ValueError(f"Unsupported pack quantization: {quant}")

    # The wrapper owns `language_model`; `modules` paths are written relative to it.
    model = q35.Model(q35.ModelArgs.from_dict({**config, "model_type": "qwen3_5"}))
    # mx.load's return type covers every container it can read; a safetensors file is
    # always the flat name->array mapping.
    weights = cast(dict[str, mx.array], mx.load(str(path / "model.safetensors")))
    prefix = "language_model." if any(
        k.startswith("language_model.") for k in weights
    ) else ""

    seen: set[str] = set()
    for record in config["modules"]:
        name = prefix + record["path"]
        if name in seen:
            raise ValueError(f"Duplicate packed module: {name}")
        seen.add(name)
        if record["dtype"] != "float16":
            raise ValueError(f"Unsupported activation dtype: {record['dtype']}")
        block = record["block"]
        if block and block not in _BLOCKS:
            raise ValueError(f"Unsupported block size: {block}")
        arrays = [weights[name + "." + s] for s in ("weight", "scales", "biases")]
        signs = weights.get(name + ".signs")
        if block and signs is None:
            raise ValueError(f"Missing sign vector for {name}")

        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        _validate(getattr(parent, parts[-1]), record, arrays, signs)
        setattr(
            parent,
            parts[-1],
            Packed(
                arrays,
                block if rotate else 0,
                signs if rotate else None,
                record["embedding"],
                _COMPUTE_DTYPE,
            ),
        )

    # Sign vectors are dropped from the weight list when the control path is loading,
    # because those modules no longer declare a `signs` parameter to receive them.
    items = [
        (k, v) for k, v in weights.items()
        if rotate or not k.endswith(".signs")
    ]
    model.load_weights(items, strict=True)
    _cast_compute_dtype(model)
    model.eval()
    mx.eval(model.parameters())
    info = {
        "packed_modules": len(seen),
        "rotated": sum(1 for r in config["modules"] if r["block"]) if rotate else 0,
        "block": next((r["block"] for r in config["modules"] if r["block"]), 0),
    }
    return model, info
