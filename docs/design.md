# Design & internals

*Why chad is built the way it is. For install and usage see the
[README](../README.md); for measured numbers see [benchmarks](benchmarks.md).*

## Why chad exists

Two ideas hold this design up.

Context is not free, and prefill is the bill. That is the engine half, and the reason
chad owns its inference loop instead of talking to a server.

The model already knows more than the harness can teach it. That is the harness half,
and the reason 2.0.0 is *smaller* than 1.x. A registry of ~56 behavioral levers and a much
larger bespoke tool surface were measured against a bare model-plus-shell loop, repeatedly,
and did not beat it. What shipped instead is five tools, ten result-channel behaviors, and a
shell the model learned in pretraining ([below](#why-the-tool-surface-is-five-tools)).

They meet in the same place. Every tool you add and every lever you teach is prompt tokens
the model re-reads on every turn, and prefill is what you pay for them.

### Context is not free, and prefill is the bill

Every turn of an agentic loop, the model has to *read* the entire conversation so far
before it can write a single new token. That read is the **prefill**: running the
transformer forward over every token in the prompt to build the **KV cache** (the
per-token attention state the model needs to keep generating). Decoding, meaning the part
that actually emits text, is memory-bandwidth bound and roughly constant. Prefill is the
part that balloons: every step appends the model's reply, the tool call, and the tool's
output to the transcript, so a naive backend re-reads an ever-longer prompt *every step*.
That is O(n) work per step and O(n²) over a session.

Concretely, on a 24 GB M4 Pro: by step 20 a real coding session is ~5,000 tokens of
transcript, which the shipped 27B prefills in ~50 s (~99 tok/s, because a dense model
reads every one of its parameters for every token of the prompt). Re-reading that every step
is most of a minute of dead air before the model says anything, and it grows faster than
linearly, since the prefill *rate* also falls as the prompt lengthens. Over a 40-step task,
prefill rather than generation is where the hours vanish.

chad's answer is a **persistent prefix KV cache**: keep the KV state alive across turns
and diff each new prompt against what's already cached, so you only prefill the handful
of *appended* tokens.

```
step N prompt:  [ system + tools | cwd · CLAUDE.md | turn 1 | … | turn N-1 | turn N ]
                └──────────────── already in the KV cache ───────────────┘ └─ new ─┘
                          prefill 0 tokens (reused verbatim)              prefill ~30
```

Same session, a couple of dozen new tokens per step instead of 5,000: **under a second of
prefill per step instead of ~50 s** (measured warm step: ~0.75 s for 16 appended tokens).
That ~67× gap is why a 27B model on a laptop answers in seconds, and it *widens*
with the transcript, since the cache-less side grows while the
warm step stays flat. The numbers are in
[benchmarks](benchmarks.md#the-agentic-loop-win-075-s-per-step-not-50-s).

### Why prefill is *hard* as well as expensive

The cache only helps if the new prompt is a strict *extension* of the cached one. Two
things fight that, and chad handles both:

- Compaction. Long sessions overflow the context window, so old tool output must be
  trimmed, which changes the prefix and would normally throw the whole cache away. chad
  compacts oldest-first and reclaims enough in a single pass that it won't re-trigger next
  step (see [Context window](configuration.md#context-window-agentic-coding-needs-room)).
- A non-trimmable cache. The shipped model is a hybrid SSM/attention model: its recurrent
  layers carry state that *can't be rewound to an arbitrary earlier token*. The cache can
  only grow by append, and any divergence forces a full rebuild. chad leans into that. It
  reuses by extension and keeps a disk-checkpointed copy of the stable system+tools prefix,
  so even a divergence reloads that ~3k-token base instead of re-prefilling it from scratch.

Everything below is how that gets built, plus the rest of what it takes to make a small model
act like a coding agent.

## Why there's no model picker

Every other local-agent harness leads with a model menu: 75-provider matrices, Ollama pulls,
quant pickers. chad ships exactly one model and no flag to change it, for three reasons:

1. The engine is fitted to the model, and that fit is the product. The DFlash2
   drafter reads this checkpoint's residual stream at five specific layers; the fused
   quantized-KV attention kernel covers this attention shape; the small-M verify matmul
   is probed per weight shape at load; the context governor knows this model's bytes per
   token. Point the same engine at arbitrary weights and every one of those is either
   absent or wrong, which is why `--model` runs slower without breaking.
2. The engine co-design doesn't survive a server boundary. The persistent prefix KV
   cache diffs *token ids* against a live cache object, so it owns the tokenizer, the
   cache layout, and the model's hybrid SSM/attention non-trimmability trade
   ([above](#why-prefill-is-hard-as-well-as-expensive)). "Just let me pick a GGUF"
   means "run through a stateless server instead," and what that costs on the same
   weights is measured in [the stock-engine comparison](benchmarks.md#same-model-same-mac-stock-engine).
3. Zero decisions is the UX. The target user comes from Claude Code, which also has no
   model picker. One command and it works, and every menu before the first task is a
   place to lose someone.

The escape hatches exist and are honest about what they cost: `--model <repo or
local dir>` forces specific weights through the same in-process engine (you keep the
cache, you lose the tuning fit), and `--backend llama` runs the harness against a remote
llama.cpp server as a measured ablation arm (you lose the on-disk warm-prefix checkpoint,
since the KV lives in the server; documented in-code).

chad ships Qwen3.8-27B on every machine, with no RAM tier and no size shorthands
(`--model` takes a repo id or a directory and nothing else). The tier that 1.x carried
existed to serve Macs below 24 GB, and chad no longer claims to: 24 GB is the target and the
floor.

## Trimmable vs. append-only: the cache trade chad lives with

The whole prefill story above hinges on reusing the KV cache. There's a second property of
a KV cache that decides *how* you're allowed to reuse it, and it's worth naming because
chad's model gives one up: **trimmability**, the ability to rewind the cache to an arbitrary
earlier token and keep going from there.

A **pure-attention** transformer is trimmable. Each token's K/V is computed independently
and stored in its own row, so "rewind to token *k*" is just "discard the rows past *k*."
That cheap rewind unlocks two things that matter on a laptop:

- Prompt-lookup / speculative decoding (PLD). Propose a draft continuation (an n-gram
  the model is about to re-quote from context), verify the whole run in one batched
  forward, and on a partial reject *roll the cache back* to the last accepted token. That
  rollback *is* a trim. With no trim, every rejected draft costs a full re-prefill. On
  novel-text-heavy generation that re-feed overhead makes the hybrid path measurably
  *slower* than just decoding, so PLD is the wrong trade without trimmability.
- Partial reuse on divergence. When a new prompt diverges from the cache mid-stream
  (compaction trimmed an old tool output, or an edit changed something in the middle) a
  trimmable cache keeps the common prefix and re-prefills only from the divergence point.
  Append-only can't: any divergence is a full rebuild.

The shipped model is not trimmable. It's a hybrid SSM/attention (`qwen3_5`) model, and its
recurrent layers carry a *fixed-size running state that is a function of the entire
sequence so far*. There's no per-token row to drop, so there's nothing to rewind to;
`cache_utils.can_trim_prompt_cache` reports false and `engine._trimmable` stays off. PLD is
gated on that flag and falls back cleanly, so it can never speed up the shipped model.

The same recurrent design is what keeps the KV footprint flat (a fixed-size SSM state no
matter how long the context grows; see the
[Context window](configuration.md#context-window-agentic-coding-needs-room) table). chad trades
trimmability for a memory profile that fits comfortably in 24 GB. The job, then, is to stay
fast on an **append-only** cache, which chad does three ways:

1. Reuse by *extension* only. The normal agentic loop only ever *appends* (the model's
   reply, the tool call, the tool output) so each new prompt is a strict extension of the
   cached one and hits the cache verbatim. That's the 99% case, and it's free.
2. Compaction that protects the prefix. When the window fills, chad compacts
   oldest-first and reclaims enough in one pass that it won't re-trigger next step, keeping
   recent turns byte-identical so the cache extension still holds.
3. A disk-checkpointed stable base. The system+tools prefix (~3k tokens, the part that
   never changes turn to turn) is persisted to disk keyed by its rendered token ids, with
   the recurrent SSM state serialized (a fixed ~51 MB floor). On a cold start *or* a
   divergence that can't be reused in RAM, that base reloads with zero prefill instead
   of being rebuilt from scratch. Two checkpoints serve it: the full prefix, which a restart
   in the same project restores outright, and its project-independent head (tool schemas +
   behavioral prompt), which any directory restores before prefilling only its own
   cwd/listing/docs tail. (Before 2.0.3 the key included that tail, so a new directory
   never hit, the same volatile-string-in-the-prefix bug `benchmarks/matrix` found in
   goose.)

So where a trimmable model would lean on PLD and partial-prefix repair, chad leans on
append-only reuse plus a warm on-disk base, and gets the responsive agentic loop anyway.
Implementation lives in `engine.py` (`_trimmable`, `warm_prefix`, the prefix diff).

## Why the tool surface is five tools

chad 2.0.0 exposes exactly `bash`, `edit`, `write`, `write_todos`, and `done`. The
1.x releases carried a much larger surface (dedicated `read`/`grep`/`glob`, a line-addressed
edit family, a tree-sitter repo map, an LSP-precise symbolic layer) and a registry of ~56
behavioral levers around it. All of it was measured
against the bare loop, repeatedly, with pre-registered paired contrasts, and none of
it beat the model plus a shell: trace measurement showed the model routes its
searching through `bash` regardless of what else is on the schema (routing follows
the trained prior, and steering text does not move it), and the lever packs all
landed inside the two-bare-arm null band.

So the design leans into the route the model actually takes:

- The model already knows the unix toolbox. `rg`, `sed -n`, `wc -l` and the project's
  own test runner are all in pretraining. Every chad-specific dialect had to be taught
  in-context, which costs prompt tokens and which the model then mostly declined to use.
- The harness's knowledge lives in the result channel (`ambient.py`). Ten levers, all ON and each ablatable via `CHAD_DISABLE`, make
  the bash route more honest and more informative: a first read of a source file
  carries a one-line symbol map, an empty grep explains which pipeline stage came up
  empty, a trimmed test run keeps its failure rows verbatim, a failed edit shows the
  first character where the sent text diverges from the file, and anything the
  harness trims hands back a path to the full body instead of destroying it.
- One editor, exact-match. `edit` (old → new, unique match) is the editing
  dialect every model knows. It recovers mechanically from the two dominant near-misses
  (literal `\n` escapes, indentation drift) without ever risking a wrong edit, because each
  recovery still requires a unique match.

A sixth tool was built and measured, and did not survive it:

- Ranked retrieval was the one primitive `bash` appeared to lack. `rg` answers
  "find this exact string" and is unbeatable at it; it cannot answer "where is FHIR
  validation handled?" without the model guessing several synonymous regexes, paying
  a round trip per miss. A BM25 index over the repo (Tantivy, one document per file)
  answered that question well: on a 26-task navigation set its ranking reached
  recall@20 26/26 and MRR 0.59, so the right file was essentially always retrieved.

  It still lost its slot. On a paired agent benchmark (same model, same tasks, same corpus,
  arms differing only in whether the tool existed) success was 6/6 in both
  arms, time to the first answer-bearing result moved +1.1% (flat), and tool-result
  context went +27.6%. Discovery calls fell on the median (4.0 → 2.5) but that was
  carried by a single task; on three of six the model ran a search *and* the same greps
  it would have run anyway, and on one it had the tool and never called it. Ranking
  quality was fine; the model reaches for the shell because that is what its prior does,
  and a tool it half-adopts is pure context cost, so the tool is gone. The
  measurement is kept in `benchmarks/search/` as the record of why, with the paired rows
  under `_runs/`. It is a record rather than a live harness: `rank.py` and `measure.py`
  import the `chad.search` module that went with the tool, so they no longer run against
  this tree.

## Architecture

The code is a standard `src/` package; tests live in `tests/`:

```
src/chad/        importable package (uv installs it as the `chad` console script)
  cli.py         argument parsing + entrypoint (chad.cli:main), plus `prove`/`levers`
  agent.py       agentic loop + guardrails
  engine.py      MLX inference + persistent prefix cache
  tools.py       the five-tool surface + JSON schemas
  ambient.py     the result-channel levers
  spill.py       the disk half of every truncation (a clip is a loan, not a deletion)
  tui.py         full-screen prompt_toolkit UI
  ...            prompt, render, repomap, validate, compaction, skills, mcp, … (modular)
tests/           pytest suites (uv run pytest)
```

```
cli.py ──▶ agent.py (agentic loop + guardrails) ──▶ engine.py (MLX + persistent prefix cache)
                 │                                          │
                 ├─ tools.py (bash · edit · write · write_todos · done)
                 └─ ambient.py (what the result channel adds back)
```

- engine.py loads the model once, keeps its KV cache alive across turns, and on every
  turn diffs the new prompt against the cached token ids so it only prefills the appended
  tokens. That's why multi-step tool loops stay snappy: re-rendering the whole transcript
  each step prefills ~20-50 new tokens while 5000+ are served from cache.
- agent.py renders the conversation through the model's chat template (with tool
  schemas), streams the turn, parses tool calls, runs them, feeds results back, and loops
  until the model stops calling tools.
- tools.py holds the five-tool surface and its JSON schemas, plus the edit forgiveness
  cascade. `ambient.py` wraps the results on the way back.

## What it borrows from other agents

Small local models are flaky tool-callers, so the harness borrows from agents that
solved the same problems:

**[forge](https://github.com/antoinezambelli/forge):** a reliability layer for self-hosted tool-calling.
- Rescue parsing. Accept `<tool_call>` XML, ```json fences, *and* bare JSON
  objects. (Weaker local coders routinely emit fenced JSON instead of the templated XML.)
- Argument validation + nudge. Missing required args get a corrective message the
  model can retry against, instead of a crash.
- No-op guard. An `edit` where `old == new` is rejected with an explanation.
- Edit recovery cascade. Dogfooding showed ~1 in 6 `edit` calls missed on mechanical
  near-misses (the model emitting literal `\n`/`\t` in `old`, or indentation/trailing-ws
  drift). `tool_edit` now retries exact → escape-normalized → whitespace-flexible, each
  still requiring a *unique* target (never edits on ambiguity), and returns the closest
  line in the file on a true miss so the model self-corrects instead of looping. Guarded
  by `test_edit.py`, whose safety half asserts the converse: a wrong or ambiguous `old`
  must not change a byte.
- Loop guard. Identical tool calls counted across the whole turn, not just
  consecutively (so an alternating `sed -n A / sed -n B` cycle is caught too); 3rd
  repeat nudges, persistent looping aborts the turn cleanly instead of spinning forever.

**[opencode](https://github.com/anomalyco/opencode) `beast` prompt:** making weaker models agentic.
- Persistence. Keep going until the request is resolved; don't yield early.
- Verify by running, and "when you say you'll call a tool, actually call it."

**[OpenHarness](https://github.com/HKUDS/OpenHarness):** base prompt structure.
- Lead-with-the-answer tone, read-before-edit, don't over-engineer, and an injected
  environment section (OS/shell/cwd).
- One principle from this list 2.0.0 *inverted*: prefer dedicated tools over `bash`.
  chad has no dedicated tools left to prefer, and the prompt now says the opposite:
  `bash` is the primary tool ([above](#why-the-tool-surface-is-five-tools)).

**[deepagents](https://github.com/langchain-ai/deepagents):** "batteries included".
- Planning tool (`write_todos`). For any 2+ step task the model lays out a plan and
  marks items `in_progress`/`completed`. The scaffold keeps a small model on-track and
  acting rather than narrating.
- Workspace snapshot. The system prompt injects a listing of the project's files
  (git-tracked or globbed) so the model knows it's in a real repo and explores it. This
  is what flipped the agent from "paste a generic rewrite into chat" to "grep → read →
  edit the actual file."
- Act-via-tools + verify-before-`done`. A refactor must go read → edit → run tests;
  the `done` tool is rejected if files were changed but nothing was run to verify them.

## Architecture map

Every module in `src/chad/`, what it owns, and the tests that guard it. The map rots one
row at a time, so a PR that adds or deletes a module should add or delete its row. Test
files are listed by which ones import the module (plus, where noted, the ones that drive
it through a re-export).

| Module | Responsibility | Guarding tests |
|---|---|---|
| `__init__.py` | Package docstring and `__version__` — the string `--version` prints and the ATIF trajectory records; must match `pyproject.toml`. | `test_cli.py` |
| `agent.py` | The agentic loop and REPL: render the transcript through the chat template, stream the turn, parse tool calls, run them, feed results back, repeat. | `test_agent.py`, `test_agent_guards.py`, `test_agent_e2e.py`, `test_intent.py` |
| `ambient.py` | Ambient state for the result channel: the levers that append harness knowledge to results the model already reads, rather than adding tools. | `test_ambient.py`, `conftest.py` |
| `atif.py` | ATIF v1.7 trajectory emitter, rebuilt from `agent.messages` after each step; a pure observer armed by `CHAD_TRAJECTORY_JSON`. | `test_atif.py` |
| `base_engine.py` | The engine seam: `GenStats` plus the `BaseEngine` Protocol that `Agent` already drives, so a second backend plugs in without touching the agent loop. | `test_completion_engine.py`, `test_agent_e2e.py`, `test_cli.py` |
| `bench.py` | The throughput benchmark behind `docs/benchmarks.md` (`chad-bench`): cold prefill, decode and warm-step tok/s on the real engine and public model. | `test_bench.py` |
| `checkpoint.py` | Shadow-git snapshots of the workspace before each file-mutating tool, in their own GIT_DIR, so `/undo` and `/restore` can revert an auto-approved edit. | `test_checkpoint.py` |
| `cli.py` | Argument parsing and entrypoint (`chad.cli:main`), plus the `prove` and `levers` subcommands. | `test_cli.py`, `test_cli_modes.py` |
| `compaction.py` | Context compaction for long sessions: every pass marks what it trimmed in-band and the notice names the spill file holding the original. | `test_compaction.py`, `test_skills.py` |
| `completion_engine.py` | The one remote backend (`--backend llama`): llama.cpp's native `/completion` endpoint driven with token-id prompts and real cache telemetry. | `test_completion_engine.py` |
| `config.py` | Single source of truth for `CHAD_*` configuration — typed accessors whose lenient parse warns and degrades a bad value to the default instead of raising. | `test_config.py`, `test_seatbelt.py` |
| `diag.py` | The opt-in diagnostic session log (`CHAD_SESSION_LOG`): throughput numbers, tool args and result previews, secret-redacted and size-rotated, never model-facing. | `test_log_redaction.py` |
| `engine.py` | The MLX inference engine and its persistent prefix KV cache, extended across turns by diffing token ids so only the newly appended tokens prefill. | `test_engine.py`, `test_engine_dflash.py`, `test_engine_kvquant.py`, `test_engine_pld_hybrid.py`, `test_engine_pld_wide.py` |
| `guardrails.py` | The pure decision predicates `run_turn` calls: loop guard, verify-before-done and empty-done gating, tool-result bookkeeping, no-tool-call nudge selection. | `test_agent_guards.py`, `test_gate.py`, `test_agent_e2e.py` |
| `ignore.py` | Single source of truth for directories no tree-walk enters (`IGNORE_DIRS`, plus `REPOMAP_EXTRA` for the repo-analysis path). | `test_ignore.py` (through the `tools`/`repomap`/`skills` re-exports) |
| `levers.py` | The registry of shipped bash-route levers, all ON by default, each keeping a name and an `enabled()` guard so `CHAD_DISABLE=a,b` can ablate one at a time. | `test_levers.py`, `test_ambient.py` |
| `mcp.py` | MCP client: read `.mcp.json`/`~/.chad/mcp.json`, connect each server over the SDK transport, expose its tools as `mcp__<server>__<tool>`, dispatch calls. | `test_mcp.py`, `test_mcp_oauth.py` |
| `mcp_oauth.py` | OAuth for hosted HTTP MCP servers: 0600 per-server token storage plus the browser/loopback redirect flow, behind the `CHAD_MCP_OAUTH` flag. | `test_mcp_oauth.py`, `test_mcp.py` |
| `mlx_dflash.py` | The DFlash2 block-diffusion drafter — load, forward, and the target-residual tap — that proposes a whole block of tokens in one drafter forward. | `test_engine_dflash.py`, `test_cli.py` |
| `mlx_fastpath.py` | Decode fast path for the dense qwen3_5 hybrid: per-row-exact weight concats and compiled S==1 layer steps that remove dispatch-bound kernel launches. | `test_mlx_fastpath.py`, `test_engine_dflash.py` |
| `mlx_qmm_mma.py` | Small-M quantized matmul for speculative verify: an MMA kernel that reads each weight group once for all rows, where stock GEMV re-pays the read per row. | `test_mlx_qmm_mma.py` |
| `mlx_qsdpa.py` | JIT-compiled fused attention over the 8-bit group-64 quantized KV cache, serving decode, speculative verification and prefill. | `test_mlx_qsdpa.py`, `test_engine_kvquant.py` |
| `prompt.py` | System-prompt construction (static prompt, workspace snapshot, project instructions) and the answer-on-paper / verify-nudge intent classifier. | `test_intent.py`, `test_warm_prefix_tiers.py`, `test_ambient.py`, `test_skills.py` |
| `prove.py` | `chad prove` — a two-minute smoke test pinned to the shipped model, offline-guarded after the cache check, reporting your own machine's numbers. | `test_prove.py` |
| `render.py` | Terminal rendering: raw tokens and tool results into a compact activity view, behind the `_emit(kind, text)` callback the REPL and the TUI each supply. | `test_render.py`, `test_confirm_preview.py`, `test_feel_pack.py` |
| `repomap.py` | Tree-sitter tag extraction for the ambient levers: language detection, mtime-cached per-file definitions, and cross-file definition lookup. | `test_repomap.py`, `test_repomap_polyglot.py`, `test_ambient.py` |
| `seatbelt.py` | macOS Seatbelt confinement for yolo-mode bash: the spawned shell (never the chad process, which needs Metal) is denied writes outside the workspace. | `test_seatbelt.py` |
| `session.py` | Conversation persistence per project directory, so `--continue`, `--resume` and the TUI `/resume` picker survive across runs. | `test_session.py`, `test_cli_modes.py`, `conftest.py` |
| `skills.py` | Agent Skills: discover `SKILL.md` dirs, parse frontmatter leniently, offer each as a slash command, and load only the one the user asks for as a user turn. | `test_skills.py`, `test_validate.py`, `test_ignore.py` |
| `speech.py` | All-local speech I/O for the TUI — Parakeet-on-MLX dictation and macOS `say` replies — with the heavy audio/MLX imports deferred to first use. | `test_speech.py`, `test_speech_tui.py` |
| `spill.py` | Spill files: every truncation writes the dropped body to disk first and the notice names the path, so a clip is a loan rather than a deletion. | `test_spill.py`, `test_compaction.py`, `test_intent.py` |
| `syntaxgate.py` | The post-mutation syntax warning: a landed edit/write that *newly* breaks a file's parse says so in the same result the model is about to read. | `test_syntaxgate.py` (through `tools.tool_edit`/`tool_write`) |
| `toolcall_parse.py` | The pure boundary between raw model text and tool dispatch: parse every tool-call dialect local models emit, de-duplicated. | `test_toolcall_parse.py`, `test_toolcall_dialect.py` |
| `tools.py` | The tool surface — `bash`, `edit`, `write`, `write_todos`, `done` — with its JSON schemas and the edit forgiveness cascade. | `test_tools.py`, `test_edit.py`, `test_gate.py`, `test_syntaxgate.py` |
| `tui.py` | The full-screen prompt_toolkit UI: mode cycling, type-ahead queue, interrupt, inline approval and status line, printed into the terminal's normal scrollback. | `test_tui.py`, `test_feel_pack.py`, `test_render.py`, `test_speech_tui.py` |
| `validate.py` | Typed tool-call validation and self-repair over `tools.SCHEMAS`: lenient parse, coerce, validate, then feedback naming exactly which fields were wrong. | `test_validate.py`, `test_tools.py`, `test_mcp.py`, `test_toolcall_dialect.py` |

### Session file format

```
~/.chad/sessions/<cwdhash>/<session_id>.json   one file per session
~/.chad/sessions/<cwdhash>/index.json          {title, updated, turns} per session id
```

A session file is a JSON object with five keys: `cwd` (absolute), `session_id`
(`YYYYMMDD-HHMMSS-<4 hex>`, minted at `Agent` construction), `updated` (epoch seconds),
`meta`, and `messages`. Only the message list is persisted — never the KV cache — so a
resume re-prefills the restored transcript. Each save overwrites only its own session file
and refreshes that session's `index.json` entry, so the `/resume` picker lists a directory
without opening every session, and resuming (which mints a fresh id and seeds the old
messages) is implicitly a fork. Known-prefix secrets are masked in tool results and
tool-call arguments on the way to disk; the in-memory transcript keeps the real values. A
legacy single-slot `~/.chad/sessions/<cwdhash>.json` is adopted as one session the first
time that directory is listed. Every write is best-effort: failing to save never breaks a
turn.

### Tool-call wire format

`toolcall_parse.parse_tool_calls` strips `<think>` blocks, then tries these dialects in
order and de-duplicates what it finds. The model's own dialect — what the chat template
renders and what the tools are described in — is the `<tool_call>` wrapper; the rest exist
because weaker local models reach for the others.

1. The XML function-call dialect `<function=name><parameter=key>value</parameter></function>` (Qwen3 / GLM thinking models); it wins when present.
2. The hybrid `{"name": "x"}` followed by `<parameter=…>` blocks — only when real parameter blocks are in scope, so a plain JSON call is left to the JSON path.
3. `<tool_call>{"name": …, "arguments": {…}}</tool_call>`.
4. A fenced block: ` ```json ` or ` ```tool_call `.
5. If nothing above matched, a closed `<tool_call>` whose JSON never closed (XML cruft stripped), and finally any bare top-level JSON object in the text.

A body that will not parse is repaired rather than dropped, because a dropped call leaves
the model with no result and no idea why: `validate.repair_json` strips the fence, fixes
trailing commas, Python constants and bare keys outside string literals, then closes
unterminated strings and brackets. A whole `arguments` value double-encoded as a string is
re-parsed the same way. What survives goes through `validate.coerce_and_validate` against
the same `tools.SCHEMAS` dict the model was sent, and any errors become
`validate.render_repair` feedback that annotates the model's own arguments, so it repairs
the marked fields instead of regenerating blindly.
