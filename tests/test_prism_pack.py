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


# --- the decode fast-path on a Prism pack ------------------------------------

from mlx_lm.models import cache as cache_utils  # noqa: E402
from mlx_lm.models.qwen3_5 import Model, ModelArgs  # noqa: E402

from chad import mlx_fastpath, mlx_qmm_mma  # noqa: E402

TINY_CFG = {
    "model_type": "qwen3_5",
    "text_config": {
        "model_type": "qwen3_5",
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "full_attention_interval": 2,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 4,
        "num_experts": 0,
        "intermediate_size": 128,
        "vocab_size": 256,
        "tie_word_embeddings": False,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"type": "default", "rope_theta": 10000.0,
                            "partial_rotary_factor": 0.25},
    },
}
BLOCK = 128
PROMPT = [3, 141, 59, 26, 53, 58, 97, 93, 23, 84, 62, 64, 33, 83, 27, 95]


def build_tiny_prism(seed=0, share_signs=True):
    """A tiny dense qwen3_5 rotated and 2-bit-g128 quantized into Packed modules the
    way the pack stores them: rows of every projection (and of the embedding table)
    put through the same transform the activations get, one sign vector per input
    width shared by every module of that width (or a fresh one each when
    `share_signs` is False — the un-fusable shape)."""
    mx.random.seed(seed)
    model = Model(ModelArgs.from_dict(TINY_CFG))
    Packed = prism_pack._packed_class()
    signs: dict = {}

    def sv(width):
        if width not in signs or not share_signs:
            signs[width] = mx.where(mx.random.uniform(shape=(width,)) > 0.5, 1.0, -1.0)
        return signs[width]

    def pack(lin, embedding=False):
        w = lin.weight
        s = sv(int(w.shape[1]))
        wq, sc, bi = mx.quantize(prism_pack.rotate(w, BLOCK, s), group_size=128, bits=2)
        return Packed(wq, sc.astype(mx.bfloat16), bi.astype(mx.bfloat16), BLOCK, s,
                      embedding, mx.bfloat16)

    lm = model.language_model
    swaps = [(lm.model, "embed_tokens", pack(lm.model.embed_tokens, True)),
             (lm, "lm_head", pack(lm.lm_head))]
    for layer in lm.model.layers:
        for name in ("gate_proj", "up_proj", "down_proj"):
            swaps.append((layer.mlp, name, pack(layer.mlp[name])))
        if layer.is_linear:
            for name in ("in_proj_qkv", "in_proj_z", "out_proj"):
                swaps.append((layer.linear_attn, name, pack(layer.linear_attn[name])))
        else:
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                swaps.append((layer.self_attn, name, pack(layer.self_attn[name])))
    model.set_dtype(mx.bfloat16)
    for owner, name, packed in swaps:
        owner[name] = packed
    model.eval()
    mx.eval(model.parameters())
    return model


def greedy(model, prompt_ids, n=6):
    cache = cache_utils.make_prompt_cache(model)
    logits = model(mx.array(prompt_ids, dtype=mx.uint32)[None], cache=cache)
    mx.eval([c.state for c in cache])
    prefill_logits = logits[0, -1].astype(mx.float32)
    cur = mx.argmax(prefill_logits)
    mx.eval(cur, prefill_logits)
    out, step_logits = [int(cur)], []
    for _ in range(n):
        logits = model(mx.array([out[-1]], dtype=mx.uint32)[None], cache=cache)
        l0 = logits[0, -1].astype(mx.float32)
        t = mx.argmax(l0)
        mx.eval(t, l0)
        step_logits.append(l0)
        out.append(int(t))
    return out, prefill_logits, step_logits


def _rel(a, b):
    return float(mx.abs(a - b).max()) / max(float(mx.abs(a).max()), 1e-6)


def test_fastpath_installs_on_a_prism_pack_and_keeps_greedy():
    model = build_tiny_prism()
    ref_ids, ref_pre, ref_steps = greedy(model, PROMPT)

    assert mlx_fastpath.install(model) is True
    lm = model.language_model
    assert hasattr(lm.model.layers[0].mlp, "_fused_signs")
    assert hasattr(lm.model.layers[0].linear_attn, "_fused_ba")
    assert hasattr(lm.model.layers[1].self_attn, "_fused_w")
    assert lm.model.layers[0]._gdn_fast is not None

    got_ids, got_pre, got_steps = greedy(model, PROMPT)
    assert _rel(ref_pre, got_pre) < 5e-2, "prefill (uncompiled, fused) drifted"
    assert got_ids == ref_ids
    worst = max(_rel(a, b) for a, b in zip(ref_steps, got_steps))
    assert worst < 5e-2, f"decode (compiled) logits drifted {worst}"


def test_fastpath_declines_a_pack_whose_projections_do_not_share_signs():
    """Fusing gate|up (or q|k|v) behind ONE rotation is only right when they were
    rotated with the same sign vector. A pack that rotates each module with its own
    must stay on the per-module Packed path — and still decode identically."""
    model = build_tiny_prism(share_signs=False)
    ref_ids, ref_pre, _ = greedy(model, PROMPT)
    assert mlx_fastpath.install(model) is False
    got_ids, got_pre, _ = greedy(model, PROMPT)
    assert got_ids == ref_ids and float(mx.abs(ref_pre - got_pre).max()) == 0.0


def test_mma_probe_sees_packed_and_fused_ternary_shapes():
    """The verify kernel's probe must find a Prism pack's projections (Packed, not
    QuantizedLinear) and the fast-path's fused stacks, keyed with the pack's g128;
    the rotated embedding table is a lookup and must not be probed. The size gate
    (N_MIN, K % 512) is for real shapes, so it is lifted for the tiny model."""
    model = build_tiny_prism()
    mlx_fastpath.install(model)
    assert mlx_qmm_mma._eligible_groups(model) == {}     # size-gated as shipped
    groups = mlx_qmm_mma._eligible_groups(model, admit=lambda K, N, bits, gs: True)
    assert all(bits == 2 and gs == 128 and K == 128 for (K, _, bits, gs) in groups)
    # fused gate|up (2 layers) + lm_head are the (128 -> 256) shapes; the embedding
    # table has the same (256, 128) footprint and must be absent.
    assert len(groups[(128, 256, 2, 128)]) == 3
    assert (128, 384, 2, 128) in groups          # fused qkv|z and fused q|k|v
    assert len(groups[(128, 128, 2, 128)]) == 4  # down x2, out_proj, o_proj
