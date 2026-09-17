"""prism_pack: the loader for Prism ML's Hadamard-folded ternary packs.

The failure these guard against is silence. A Prism pack's weights, scales and
biases have exactly the shapes mlx-lm's ordinary affine loader expects, so the
wrong loader produces a model that runs and returns plausible garbage instead of
raising. So the routing predicate and the container check are pinned here, and
the rotation itself is checked against a hand-computed Hadamard round-trip on a
tiny synthetic pack — no 8.6 GB download.
"""

import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from chad import prism_pack  # noqa: E402


def test_is_prism_pack_routes_on_declared_type():
    assert prism_pack.is_prism_pack({"model_type": "prism_hadamard_qwen35"})
    assert not prism_pack.is_prism_pack({"model_type": "qwen3_5"})
    assert not prism_pack.is_prism_pack({})


def test_load_refuses_a_foreign_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    with pytest.raises(ValueError, match="not a Prism pack"):
        prism_pack.load(str(tmp_path))


@pytest.mark.parametrize("quant", [
    {"bits": 4, "group_size": 128, "mode": "affine"},   # a width we never wrote
    {"bits": 2, "group_size": 64, "mode": "affine"},    # g64, not the pack's g128
    {"bits": 2, "group_size": 128, "mode": "mxfp4"},    # a different container
    {},                                                  # undeclared
])
def test_load_refuses_an_unsupported_container(tmp_path, quant):
    """The rotation is only correct for the container it was written against, and a
    mismatch would decode silently rather than raise — so refuse before building."""
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "prism_hadamard_qwen35", "quantization": quant,
        "text_config": {}, "modules": []}))
    with pytest.raises(ValueError, match="unsupported Prism container"):
        prism_pack.load(str(tmp_path))


def test_packed_linear_applies_the_rotation():
    """A rotated Packed must equal fwht(x) @ W, and must NOT equal the plain x @ W an
    ordinary affine loader would compute — the whole reason this module exists."""
    Packed = prism_pack._packed_class()
    rows, width, block = 256, 128, 128
    w = mx.random.normal((rows, width))
    wq, scales, biases = mx.quantize(w, group_size=128, bits=2)
    signs = mx.where(mx.random.uniform(shape=(width,)) > 0.5, 1.0, -1.0)
    x = mx.random.normal((1, 3, width)).astype(mx.float16)

    packed = Packed(wq, scales, biases, block, signs, False, mx.float16)
    rotated = mx.hadamard_transform((x.astype(mx.float32) * signs).reshape(-1, block),
                                    scale=1 / math.sqrt(block)).reshape(x.shape)
    want = mx.quantized_matmul(rotated.astype(mx.float16), wq, scales, biases,
                               transpose=True, group_size=128, bits=2)
    unrotated = mx.quantized_matmul(x, wq, scales, biases, transpose=True,
                                    group_size=128, bits=2)

    assert mx.allclose(packed(x), want, atol=1e-2).item()
    assert not mx.allclose(packed(x), unrotated, atol=1e-2).item()


def test_packed_embedding_inverts_the_rotation():
    """Embedding rows are stored rotated, so the lookup applies the INVERSE transform:
    signs after the Hadamard, not before it."""
    Packed = prism_pack._packed_class()
    vocab, width, block = 64, 128, 128
    w = mx.random.normal((vocab, width))
    wq, scales, biases = mx.quantize(w, group_size=128, bits=2)
    signs = mx.where(mx.random.uniform(shape=(width,)) > 0.5, 1.0, -1.0)
    ids = mx.array([[1, 7, 63]], dtype=mx.int32)

    packed = Packed(wq, scales, biases, block, signs, True, mx.float16)
    rows = mx.dequantize(wq[ids.reshape(-1)], scales[ids.reshape(-1)],
                         biases[ids.reshape(-1)], group_size=128, bits=2)
    want = mx.hadamard_transform(rows.reshape(-1, block).astype(mx.float32),
                                 scale=1 / math.sqrt(block)).reshape(1, 3, width) * signs

    assert packed(ids).shape == (1, 3, width)
    assert mx.allclose(packed(ids), want.astype(mx.float16), atol=1e-2).item()


def test_packed_without_a_block_is_a_plain_affine_matmul():
    """`block == 0` marks a tensor the pack stored unrotated; it must skip the
    transform entirely rather than apply an identity-shaped one."""
    Packed = prism_pack._packed_class()
    w = mx.random.normal((256, 128))
    wq, scales, biases = mx.quantize(w, group_size=128, bits=2)
    x = mx.random.normal((1, 3, 128)).astype(mx.float16)

    packed = Packed(wq, scales, biases, 0, None, False, mx.float16)
    want = mx.quantized_matmul(x, wq, scales, biases, transpose=True,
                               group_size=128, bits=2)
    assert mx.allclose(packed(x), want).item()
