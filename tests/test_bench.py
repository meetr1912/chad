"""Tier-1 tests for chad-bench's CLI wiring (bench.py). No model load.

Regression origin: a dogfood replace_symbol rewrite of `main` silently dropped the
--context-tokens argparse line while _run_agentic still read args.context_tokens, so
`chad-bench --agentic` crashed with AttributeError. The rewrite parsed cleanly — the
edit gate can't see semantic drift — so the arg↔consumer contract gets a real test.
"""
import os

from chad import bench


def _local_model(monkeypatch, tmp_path):
    """A local model dir named by CHAD_MODEL: the real model resolution takes it as the
    override, and there is nothing to download for a directory."""
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setenv("CHAD_MODEL", str(model))
    return str(model)


def test_agentic_wires_context_tokens(monkeypatch, tmp_path):
    """`chad-bench --agentic` must reach _run_agentic with the parsed --context-tokens
    (default 24000) — the exact call that regressed."""
    model = _local_model(monkeypatch, tmp_path)
    seen: dict = {}

    def run_agentic(mid, why, ctx):
        seen.update(mid=mid, ctx=ctx)
        return 0

    assert bench.main(["--agentic"], run_agentic=run_agentic) == 0
    assert seen == {"mid": model, "ctx": 24000}
    seen.clear()
    assert bench.main(["--agentic", "--context-tokens", "12000"],
                      run_agentic=run_agentic) == 0
    assert seen["ctx"] == 12000


def test_chunk_arg_exports_env(monkeypatch, tmp_path):
    """--chunk reaches the engine via CHAD_PREFILL_CHUNK before any benchmark runs.

    bench.main writes os.environ directly, so clean up with save/restore — a
    monkeypatch.delenv AFTER the write would make teardown restore the leaked value
    (delenv records the current value to put back), poisoning later engine tests."""
    _local_model(monkeypatch, tmp_path)
    prior = os.environ.pop("CHAD_PREFILL_CHUNK", None)
    try:
        assert bench.main(["--agentic", "--chunk", "128"],
                          run_agentic=lambda mid, why, ctx: 0) == 0
        assert os.environ.get("CHAD_PREFILL_CHUNK") == "128"
    finally:
        if prior is None:
            os.environ.pop("CHAD_PREFILL_CHUNK", None)
        else:
            os.environ["CHAD_PREFILL_CHUNK"] = prior
