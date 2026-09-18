"""Split the decode-speed deficit into its two candidate causes.

The rotated pack decodes at 11 tok/s where the 3-bit build does 44. Two things differ:
the transforms themselves (extra kernels per projection), and chad's fused/compiled
decode path, which does not recognise the rotated modules and silently skips.

Loading the same weights with `rotate=False` removes the transforms and changes nothing
else — wrong numerically, but it isolates their cost. A third run compiles the whole
model step, which is the ceiling a rotation-aware fastpath could reach.

    uv run python prism_profile.py
"""

import time

import mlx.core as mx

from chad.mlx_prism import load


def decode_tps(model, steps=64, warmup=4):
    ids = mx.array([[1] * 16], dtype=mx.int32)
    cache = model.language_model.make_cache()
    logits = model(ids, cache=cache)
    tok = mx.argmax(logits[:, -1, :], axis=-1)[None]
    mx.eval(tok)
    for _ in range(warmup):
        tok = mx.argmax(model(tok, cache=cache)[:, -1, :], axis=-1)[None]
    mx.eval(tok)
    t0 = time.time()
    for _ in range(steps):
        tok = mx.argmax(model(tok, cache=cache)[:, -1, :], axis=-1)[None]
    mx.eval(tok)
    return steps / (time.time() - t0)


def main() -> None:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        "nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX", local_files_only=True
    )
    for rotate in (True, False):
        model, _ = load(path, rotate=rotate)
        tps = decode_tps(model)
        print(f"rotation {'ON ' if rotate else 'OFF'}  {tps:5.1f} tok/s")
        del model
        mx.clear_cache()


if __name__ == "__main__":
    main()
