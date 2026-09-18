"""Discriminating check for the Prism Hadamard loader.

"It loads" and "it emits text" are both consistent with the rotation being skipped —
that is the whole hazard. This measures teacher-forced NLL on a code sample with the
rotation ON and OFF against the same weights and the same loader. Correct reconstruction
lands ~1-2.5 nats/token on ordinary code; an unrotated basis lands in the high single
digits and is fluent-looking nonsense when sampled.

    uv run python prism_nll.py [model_dir]
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from chad.mlx_prism import load

SAMPLE = '''def binary_search(items, target):
    lo, hi = 0, len(items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
'''


def nll(model, ids):
    x = mx.array([ids[:-1]], dtype=mx.int32)
    logits = model(x).astype(mx.float32)
    targets = mx.array([ids[1:]], dtype=mx.int32)
    losses = nn.losses.cross_entropy(logits, targets, reduction="none")
    mx.eval(losses)
    return float(mx.mean(losses).item())


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else _default_path())
    from mlx_lm.utils import load_tokenizer

    ids = load_tokenizer(path).encode(SAMPLE)
    print(f"{len(ids)} tokens from {path.name}")

    results = {}
    for rotate in (True, False):
        model, info = load(str(path), rotate=rotate)
        results[rotate] = nll(model, ids)
        label = "rotation ON " if rotate else "rotation OFF"
        print(f"{label}  NLL {results[rotate]:7.3f} nats/token  "
              f"({info['rotated']} rotated modules, block {info['block']})")
        del model
        mx.clear_cache()

    on, off = results[True], results[False]
    print(f"\ngap: {off - on:+.3f} nats/token")
    if on >= off:
        print("FAIL: rotation did not improve the fit — transform order is wrong")
        return 1
    if on > 4.0:
        print(f"FAIL: rotated NLL {on:.3f} is too high for code; reconstruction is off")
        return 1
    print("PASS: rotation reconstructs the weights")
    return 0


def _default_path() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        "nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX",
        local_files_only=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
