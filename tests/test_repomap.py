"""Tests for the tree-sitter repo map (repomap.py), focused on the file-scan
memoization in the repo-map ranker.

`RepoMap._code_files()` walks the whole tree (pruned `os.walk`) plus an isfile/
getsize/language-detect on every entry, and it runs on *every* find_symbol /
view_symbol / find_refs / repo_map call. The per-file parse is already mtime-cached
(`_extract`), but the directory walk was not — so an agent doing a handful of symbol
lookups re-walked the entire repo each time. Since `repomap.service()` is a cwd-cached
singleton that lives for the whole session, memoizing the file list on the instance
lets every lookup after the first reuse it.

This file pins three properties:

  1. Correctness — the cache doesn't break lookups (view_symbol/find_symbol still
     return the expected symbol).
  2. Single-scan — two symbol lookups glob the tree exactly once, not once per call.
  3. No-cache-on-interrupt — a scan stopped mid-walk by `should_stop` is NOT cached,
     so a later normal call still computes the full list.

No model is loaded; this runs in the fast gate.
"""

import json
import os
import pickle
import subprocess
import sys
import time

import pytest

from chad import repomap
from chad.repomap import RepoMap

passed = 0
failed = 0


def check(desc, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"PASS: {desc}")
    else:
        failed += 1
        raise AssertionError(f"{desc}{(' — ' + detail) if detail else ''}")


def _make_repo(tmp_path):
    """A tiny fixture repo with two .py files and known symbols."""
    (tmp_path / "alpha.py").write_text(
        "def alpha_func(x):\n"
        "    return x + 1\n"
        "\n"
        "class AlphaClass:\n"
        "    def method_one(self):\n"
        "        return alpha_func(1)\n"
    )
    (tmp_path / "beta.py").write_text(
        "from alpha import alpha_func\n"
        "\n"
        "def beta_func():\n"
        "    return alpha_func(2)\n"
    )
    return str(tmp_path)


def test_disk_cache(tmp_path=None):
    """The persistent tags cache: a second RepoMap instance (a new session) serves
    the scan from disk without re-parsing, and an mtime change re-parses that file."""
    import pathlib
    import tempfile
    if tmp_path is None:
        tmp_path = pathlib.Path(tempfile.mkdtemp(prefix="repomap_cache_"))
    cache_dir = str(tmp_path / "cachedir")
    repo = tmp_path / "repo"
    repo.mkdir()
    # enough files to clear the save threshold
    for i in range(repomap._CACHE_SAVE_MIN + 2):
        (repo / f"mod{i:02d}.py").write_text(f"def func_{i:02d}():\n    pass\n")

    rm1 = RepoMap(str(repo), cache_dir=cache_dir)
    files = rm1._code_files()
    rm1._extract_all(files)
    check("first scan sees the fixture symbols",
          bool(rm1._find_defs("func_00")))
    cache_files = list(pathlib.Path(cache_dir).glob("*.pkl"))
    check("scan above the save threshold persists a cache file",
          len(cache_files) == 1, repr(cache_files))

    # A fresh instance must be served from disk: one without tree-sitter cannot parse,
    # so every tag it returns came out of the cache file.
    rm2 = RepoMap(str(repo), cache_dir=cache_dir, tree_sitter=False)
    rm2._extract_all(files)
    names = {d.name for f in files for d in rm2._extract(f)[0]}
    check("warm scan is served entirely from the disk cache",
          {"func_00", "func_33"} <= names, repr(sorted(names)))

    # mtime invalidation: rewrite one file, its new symbol must appear.
    target = repo / "mod00.py"
    target.write_text("def renamed_func():\n    pass\n")
    os.utime(target, (1e9, 1e9))  # force an mtime change even on coarse clocks
    rm3 = RepoMap(str(repo), cache_dir=cache_dir)
    rm3._extract_all(rm3._code_files())
    check("a changed file is re-parsed (new symbol appears)",
          bool(rm3._find_defs("renamed_func")))
    check("the changed file's old symbol is gone",
          not rm3._find_defs("func_00"))


def test_parallel_extract(tmp_path=None):
    """Subprocess-sharded extraction returns the same tags as the serial path, and
    an immediate should_stop kills the workers without hanging."""
    import pathlib
    import tempfile
    if tmp_path is None:
        tmp_path = pathlib.Path(tempfile.mkdtemp(prefix="repomap_par_"))
    for i in range(24):
        (tmp_path / f"m{i:02d}.py").write_text(
            f"def par_func_{i:02d}(x):\n    return x\n")

    serial = RepoMap(str(tmp_path))
    serial._disk_checked = True  # isolate from any real on-disk cache
    for f in serial._code_files():
        serial._extract(f)

    real_min = repomap._PARALLEL_MIN_FILES
    repomap._PARALLEL_MIN_FILES = 1  # force the worker path on the tiny fixture
    try:
        par = RepoMap(str(tmp_path))
        par._disk_checked = True
        par._extract_all(par._code_files())
        same = all(par._cache.get(f, (None, [], []))[1:]
                   == serial._cache.get(f, (None, [], []))[1:]
                   for f in serial._code_files())
        check("parallel extraction matches serial tags exactly", same)

        stopped = RepoMap(str(tmp_path))
        stopped._disk_checked = True
        stopped._extract_all(stopped._code_files(), should_stop=lambda: True)
        check("should_stop during parallel extraction returns without hanging", True)
    finally:
        repomap._PARALLEL_MIN_FILES = real_min


if __name__ == "__main__":
    test_disk_cache()
    test_parallel_extract()
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)


# --- wheel-less platforms ------------------------------------------
# tree-sitter-language-pack is a native wheel. On a benchmark container (emulated amd64) uv
# fell back to a Rust source build, it failed, and the module-level import took ALL of
# chad down -- the qemu-startup trial errored before the agent ran one step.

def test_repomap_degrades_when_tree_sitter_is_absent(tmp_path):
    """With tree-sitter unavailable, symbol ranking yields nothing but nothing raises."""
    (tmp_path / "a.py").write_text("def f():\n    pass\n")
    rm = repomap.RepoMap(str(tmp_path), tree_sitter=False)
    assert rm.lang_for(str(tmp_path / "a.py")) is None
    assert rm._lang_tools("python") is None
    assert rm._extract(str(tmp_path / "a.py")) == ([], [])
    assert rm._find_defs("f") == []


def test_repomap_and_tools_import_without_the_tree_sitter_wheel(tmp_path):
    """The regression that matters: `tools` (bash/read/edit) imports `repomap` at module
    scope, so a missing tree-sitter wheel used to make ALL of chad unimportable. A child
    interpreter whose tree-sitter packages fail to import, as a broken wheel install's
    do, must still import both, and its repo map must degrade."""
    stubs = tmp_path / "no_wheel"
    stubs.mkdir()
    for pkg in ("tree_sitter", "tree_sitter_language_pack"):
        (stubs / f"{pkg}.py").write_text(
            f"raise ImportError('no wheel for {pkg} on this platform')\n")
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    code = ("import chad.tools\n"
            "from chad import repomap\n"
            "assert repomap.HAVE_TREE_SITTER is False, 'import guard did not engage'\n"
            "assert repomap.RepoMap('.').lang_for('a.py') is None\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path),
                       env={**os.environ, "PYTHONPATH": os.pathsep.join([str(stubs), src])},
                       capture_output=True, text=True, timeout=120, check=False)
    assert r.returncode == 0, r.stderr


# --- bounded lookups and the on-disk cache ----------------------------------------

def test_find_defs_without_refresh_never_rewalks(tmp_path):
    rm = RepoMap(_make_repo(tmp_path))
    rm._disk_checked = True
    assert rm._find_defs("alpha_func", refresh=False)
    assert rm._find_defs("no_such_symbol", refresh=False) == []
    # A file created after the memoized walk, once that walk is old enough to be
    # re-walked on a miss: only the default lookup may go looking for it.
    time.sleep(1.1)
    (tmp_path / "gamma.py").write_text("def gamma_func():\n    pass\n")
    assert rm._find_defs("gamma_func", refresh=False) == []
    assert rm._find_defs("gamma_func")


def _cacheable_repo(tmp_path):
    """A repo big enough to persist its tags, and a private cache dir for it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for i in range(repomap._CACHE_SAVE_MIN + 2):
        (repo / f"mod{i:02d}.py").write_text(f"def func_{i:02d}():\n    pass\n")
    return str(repo), str(tmp_path / "cache")


def test_cache_file_is_a_header_line_then_the_entries(tmp_path):
    repo, cache = _cacheable_repo(tmp_path)
    rm1 = RepoMap(repo, cache_dir=cache)
    rm1._extract_all(rm1._code_files())
    with open(rm1._cache_file(), "rb") as f:
        assert json.loads(f.readline()) == {
            "v": repomap._CACHE_VERSION, "tag_fields": list(repomap.Tag._fields),
            "root": rm1.root}
    rm2 = RepoMap(repo, cache_dir=cache)
    rm2._load_disk_cache()
    assert rm2._cache and rm2._cache == rm1._cache


class _Tripwire:
    """Unpickling one creates the directory `marker`: proof a cache body was loaded."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (os.mkdir, (self.marker,))


@pytest.mark.parametrize("stale", ["old_version", "pre_header_layout"])
def test_stale_cache_file_is_deleted_without_unpickling(tmp_path, stale):
    repo, cache = _cacheable_repo(tmp_path)
    rm = RepoMap(repo, cache_dir=cache)
    marker = str(tmp_path / "unpickled")
    fields = {"v": repomap._CACHE_VERSION, "tag_fields": list(repomap.Tag._fields),
              "root": rm.root}
    if stale == "old_version":
        blob = (json.dumps({**fields, "v": repomap._CACHE_VERSION - 1}) + "\n").encode()
        blob += pickle.dumps(_Tripwire(marker))
    else:  # one pickled dict carrying its own version, as written before the header
        blob = pickle.dumps({**fields, "v": 3, "files": _Tripwire(marker)})
    os.makedirs(cache)
    with open(rm._cache_file(), "wb") as f:
        f.write(blob)
    rm._load_disk_cache()
    assert not os.path.exists(marker)
    assert not os.path.exists(rm._cache_file())
    assert rm._cache == {}


def test_sweep_bounds_the_cache_dir_and_spares_the_current_repo(tmp_path):
    now = time.time()

    def aged(name, size, days):
        p = tmp_path / name
        p.write_bytes(b"x" * size)
        os.utime(p, (now - days * 86400,) * 2)
        return str(p)

    mine = aged("mine.pkl", 600, 90)   # past max age and the oldest: still kept
    aged("expired.pkl", 10, 40)        # past max age
    aged("older.pkl", 500, 5)          # oldest live file, evicted to fit the cap
    aged("newer.pkl", 300, 1)
    aged("notes.txt", 5000, 90)        # not a cache file
    repomap._sweep_cache_dir(str(tmp_path), mine, max_bytes=1000, max_age_s=30 * 86400)
    survivors = sorted(p.name for p in tmp_path.iterdir())
    assert survivors == ["mine.pkl", "newer.pkl", "notes.txt"]


def test_first_save_in_a_process_sweeps_once(tmp_path):
    repo, cache = _cacheable_repo(tmp_path)
    rm = RepoMap(repo, cache_dir=cache)
    os.makedirs(cache)

    def expired(name):
        p = os.path.join(cache, name)
        with open(p, "wb") as f:
            f.write(b"x")
        old = time.time() - repomap._CACHE_MAX_AGE_S - 86400
        os.utime(p, (old, old))
        return p

    first = expired("first.pkl")
    rm._save_disk_cache(rm._code_files())
    assert not os.path.exists(first)
    assert os.path.exists(rm._cache_file())   # the saving repo's own file is spared
    second = expired("second.pkl")
    rm._save_disk_cache(rm._code_files())
    assert os.path.exists(second)
