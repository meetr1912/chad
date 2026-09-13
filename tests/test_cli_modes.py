"""`cli.main` mode matrix: the permission mode, reasoning mode and resume state the
entrypoint hands to `Agent` for a given argv and terminal.

The line that matters most is the headless promotion: a one-shot task with no TTY on
stdin runs in yolo (every write and command auto-approved), because the confirm prompt
would read EOF and abort every edit. `--plan` must suppress that promotion and a TTY
must never get it — a refactor that inverts either turns no other test red.

These drive the real `_main` (real argparse, real session store — redirected to a tmp
dir by conftest) with the model-loading seams stubbed and `Agent` / `run_tui` replaced
by recorders, so no weights load and each assertion reads exactly what `_main` decided.
"""
import io
import os
import sys
from types import SimpleNamespace

import pytest

from chad import cli, session


class _FakeEngine:
    """The attributes `_main` touches on an engine, and nothing else."""

    effective_ctx = 32768
    temp = 0.0

    def __init__(self, **kw):
        self.loads = 0
        self.resets = 0

    def load(self):
        self.loads += 1
        return 0.0

    def reset(self):
        self.resets += 1


class _Stdin(io.StringIO):
    def __init__(self, tty):
        super().__init__("")
        self._tty = tty

    def isatty(self):
        return self._tty


def _tty(monkeypatch, tty):
    monkeypatch.setattr(sys, "stdin", _Stdin(tty))


@pytest.fixture
def rec(monkeypatch, tmp_path):
    """Stub the model-loading seams; record every engine, Agent and TUI launch.

    `rec.notes` scripts the `budget_note` each successive turn banks, to drive the
    relaunch loop; left empty, every turn finishes clean. The TUI is always stubbed so a
    regression that falls through to it fails an assertion instead of taking the terminal."""
    rec = SimpleNamespace(agents=[], engines=[], notes=[], tui=None)

    class _RecordingAgent:
        def __init__(self, eng, **kw):
            self.eng, self.kw, self.turns = eng, kw, []
            self.budget_note = None
            self.saved = False
            rec.agents.append(self)

        def run_turn(self, text):
            self.turns.append(text)
            self.budget_note = rec.notes.pop(0) if rec.notes else None
            return "ok"

        def save(self):
            self.saved = True

    def _engine(**kw):
        rec.engines.append(_FakeEngine(**kw))
        return rec.engines[-1]

    def _run_tui(eng, ctx_limit, **kw):
        rec.tui = kw

    monkeypatch.setattr(cli, "_preflight", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_pick_model", lambda *a, **k: ("stub-model", "test"))
    monkeypatch.setattr(cli, "_ensure_model", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_compute_ctx_limit", lambda eng: 24000)
    monkeypatch.setattr(cli, "peek_ctx_limit", lambda *a, **k: 24000)
    monkeypatch.setattr(cli, "Engine", _engine)
    monkeypatch.setattr(cli, "Agent", _RecordingAgent)
    monkeypatch.setattr("chad.engine.peek_context_window", lambda *a, **k: 32768)
    monkeypatch.setattr("chad.tui.run_tui", _run_tui)
    # The knobs that change _main's control flow on the paths under test.
    for var in ("CHAD_AUTO_CONTINUE", "CHAD_TURN_BUDGET_S", "CHAD_REVIEW_PASS",
                "CHAD_CTX_LIMIT", "CHAD_DISABLE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # the session store is keyed on cwd
    return rec


# --- one-shot permission mode -------------------------------------------------

@pytest.mark.parametrize("argv, tty, mode, promoted", [
    (["do X"], False, "yolo", True),            # headless: promoted, and says so
    (["do X"], True, "normal", False),          # a TTY keeps the confirm prompt
    (["--plan", "do X"], False, "plan", False),  # --plan is never promoted
    (["--plan", "do X"], True, "plan", False),
    (["--yolo", "do X"], True, "yolo", False),  # explicit, so no promotion banner
])
def test_one_shot_permission_mode(rec, monkeypatch, capsys, argv, tty, mode, promoted):
    _tty(monkeypatch, tty)

    cli.main(argv)

    (agent,) = rec.agents
    assert agent.kw["mode"] == mode
    assert agent.kw["yolo"] is (mode == "yolo")
    assert agent.kw["thinking"] is True
    assert agent.kw["resume"] is None
    assert agent.turns == ["do X"]
    assert agent.saved  # a follow-up `chad -c` can pick the thread up
    assert ("[headless: auto-approving" in capsys.readouterr().err) is promoted


def test_bad_argv_still_fails_in_argparse(rec, monkeypatch):
    _tty(monkeypatch, True)

    with pytest.raises(SystemExit) as exc:
        cli.main(["--no-such-flag", "do X"])

    assert exc.value.code == 2
    assert rec.agents == [] and rec.engines == []


def test_no_think_turns_thinking_off(rec, monkeypatch):
    _tty(monkeypatch, True)

    cli.main(["--no-think", "do X"])

    assert rec.agents[0].kw["thinking"] is False


# --- resume -------------------------------------------------------------------

def test_continue_resumes_the_newest_session(rec, monkeypatch):
    older = [{"role": "user", "content": "older task"}]
    newest = [{"role": "user", "content": "newest task"},
              {"role": "assistant", "content": "newest answer"}]
    session.save_session(os.getcwd(), older, {}, session_id="20260101-000000-aaaa")
    session.save_session(os.getcwd(), newest, {}, session_id="20260101-000100-bbbb")
    _tty(monkeypatch, True)

    cli.main(["-c", "do X"])

    assert rec.agents[0].kw["resume"] == newest


def test_continue_without_a_saved_session_starts_fresh(rec, monkeypatch, capsys):
    _tty(monkeypatch, True)

    cli.main(["-c", "do X"])

    assert rec.agents[0].kw["resume"] is None
    assert "no saved session" in capsys.readouterr().err


def test_resume_without_a_tty_exits_instead_of_prompting(rec, monkeypatch, capsys):
    session.save_session(os.getcwd(), [{"role": "user", "content": "t"}], {})
    _tty(monkeypatch, False)

    with pytest.raises(SystemExit) as exc:
        cli.main(["--resume"])

    assert exc.value.code == 1
    assert rec.agents == [] and rec.tui is None
    assert "needs an interactive terminal" in capsys.readouterr().err


def test_resume_on_a_tty_loads_the_picked_session_not_the_newest(rec, monkeypatch):
    older = [{"role": "user", "content": "older task"}]
    session.save_session(os.getcwd(), older, {}, session_id="20260101-000000-aaaa")
    session.save_session(os.getcwd(), [{"role": "user", "content": "newer"}], {},
                         session_id="20260101-000100-bbbb")
    _tty(monkeypatch, True)
    monkeypatch.setattr(cli, "_pick_session", lambda items: items[-1])  # newest first

    cli.main(["--resume", "do X"])

    assert rec.agents[0].kw["resume"] == older


# --- no task: the TUI ---------------------------------------------------------

@pytest.mark.parametrize("argv, mode", [
    ([], "normal"),
    (["--plan"], "plan"),
    (["--yolo"], "yolo"),
])
def test_no_task_launches_the_tui_in_the_requested_mode(rec, monkeypatch, argv, mode):
    _tty(monkeypatch, True)

    cli.main(argv)

    assert rec.agents == []  # the TUI builds its own Agent
    assert rec.tui["mode"] == mode
    assert rec.tui["thinking"] is True and rec.tui["resume"] is None
    assert rec.engines[0].loads == 0  # weights load on the TUI's thread, via finalize


# --- one-shot relaunch after a budget stop ------------------------------------

def test_headless_budget_stop_relaunches_fresh_with_the_progress_note(rec, monkeypatch):
    _tty(monkeypatch, False)
    monkeypatch.setenv("CHAD_TEMP", "0")  # greedy, so the relaunch has to raise it
    rec.notes = ["Progress so far: edited a.py", None]

    cli.main(["do X"])

    first, second = rec.agents
    assert second.kw["mode"] == "yolo" and second.kw["yolo"] is True
    assert "resume" not in second.kw  # a fresh context, not a replay of the stuck one
    assert second.turns == ["do X\n\n[Progress so far: edited a.py]"]
    assert rec.engines[0].resets == 1
    assert rec.engines[0].temp == 0.6  # a greedy stall would replay itself verbatim
    assert second.saved and not first.saved


def test_headless_relaunches_stop_after_two(rec, monkeypatch):
    _tty(monkeypatch, False)
    rec.notes = ["note 1", "note 2", "note 3", "note 4"]

    cli.main(["do X"])

    assert len(rec.agents) == 3
    assert rec.agents[-1].budget_note == "note 3"  # still banked; the run gives up
    assert rec.agents[-1].saved


def test_interactive_budget_stop_does_not_relaunch(rec, monkeypatch):
    _tty(monkeypatch, True)
    rec.notes = ["Progress so far: nothing landed"]

    cli.main(["do X"])

    assert len(rec.agents) == 1 and rec.agents[0].saved
    assert rec.engines[0].resets == 0
