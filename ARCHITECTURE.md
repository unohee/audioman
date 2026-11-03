# Architecture — `audioman`

A command-line wrapper around VST3/AU audio plugins (iZotope RX, FabFilter, and any
other installed bundle), plus the built-in measurement and mastering tooling that
surrounds them. Every subcommand has a `--json` mode, so the tool is usable as a
subprocess from an agent or a pipeline as well as by hand.

The engineering standard this repository is held to is
[`book.md`](https://github.com/unohee/dev_runbook/blob/main/book.md); CI enforces it
through `unohee/ci-templates`. Articles are cited below as `Art. III`, protocols as
`§5.5`.

## Layer model

Five packages under `src/audioman/`, each depending only on the ones to its right:

| Layer | Path | Holds |
|---|---|---|
| Presentation | `cli/` | argparse wiring, output formatting, exit handling. 26 subcommands, one module each. |
| Domain | `core/` | DSP, measurement, file I/O, session/EDL models, plugin *discovery*. Pure functions and dataclasses; no printing. |
| Plugin access | `plugins/` | The `PluginWrapper` Protocol and its pedalboard-backed implementation (`vst3.py`), plus parameter metadata types. |
| Configuration | `config/` | Platform search paths (`paths.py`) and pydantic-settings (`settings.py`). |
| GPU (optional) | `gpu/`, `core/gpu_spectral.py` | Capability probe (`gpu/__init__.py`) and torch-backed spectral/de-reverb helpers. Not on the default path; see *Invariants*. |

`core/__init__.py`, `plugins/__init__.py` and `config/__init__.py` are empty. There is
no import from `core/` back into `cli/`, and none from `plugins/` into `core/` — the
dependency direction is one-way, which is what lets the streaming layer and the plugin
analysis layer be driven from a test without a terminal.

**`core/findings.py` is the canonical fault contract.** It defines `Category`
(`signal` / `spectral` / `plugin` / `container`), `Severity`, the stable `Code` enum, and
two envelope builders (`json_envelope`, `envelope`). Every `--json` payload in the CLI
goes through it, so the shape an agent reads is the same shape a test asserts. It has no
dependencies beyond the standard library and is imported by `cli/`, `core/detectors.py`
and `core/discontinuity.py`.

## Plugin discovery, registration, invocation

Only **VST3 and AU bundles** are registrable — `PluginRegistry` globs `**/*.vst3` and
`*.component` under the platform search paths and refuses anything else. There is no
dynamically loaded plugin type; a built-in Python plugin is a `builtin:` name resolved
by the `stream` subcommand, not a registry entry.

The path is `config.paths` → `core/registry.py` → `plugins/vst3.py` → callers:

1. `config/paths.py:get_vst3_search_paths()` picks the directories per OS — macOS
   `/Library/Audio/Plug-Ins/VST3` and `~/Library/...`, Linux `/usr/lib/vst3`,
   `/usr/local/lib/vst3`, `~/.vst3`, Windows `C:/Program Files/Common Files/VST3`.
   `get_au_search_paths()` returns the macOS Components directories and is empty
   elsewhere. `AudiomanSettings.extra_vst3_paths` / `extra_au_paths` append to these.
2. `core/registry.py:PluginRegistry.scan()` walks those directories, and for each bundle
   parses `Contents/Info.plist` (`_parse_vst3_info`, `_parse_au_info`) into a
   `PluginMeta` (name, `short_name`, path, format, vendor, version, aliases). AU loses to
   VST3 on a `short_name` collision. The result is cached as
   `~/.audioman/cache/plugins.json` and reused unless `refresh=True` or every cached path
   has since disappeared. `get_registry()` is the process-wide singleton.
3. `PluginRegistry.get(name)` resolves a `short_name` first, then an alias from the
   `ALIASES` table (`denoise` → `spectral-de-noise`), and returns `PluginMeta` or
   `None`. The registry never loads the plugin binary.
4. Callers build a `VST3PluginWrapper(meta.path)` and call `load()`, which imports
   pedalboard, `load_plugin()`s the bundle, and dup2-redirects stdout/stderr around the
   call (iZotope plugins print Objective-C runtime chatter that would otherwise corrupt
   `--json` output — the redirect is serialized by `_FD_REDIRECT_LOCK`).
5. Work then goes through `plugins/base.py`'s `PluginWrapper` Protocol —
   `load` / `get_parameters` / `set_parameters` / `process(audio, sample_rate, reset)` /
   `reset` — so anything satisfying it (the wrapper, a pedalboard board via
   `core/streaming.make_pedalboard_process_fn`) can be driven by the engine, the
   pipeline, the commit path and the streaming bench.

Callers of step 3–4: `cli/{scan,list_cmd,info,dump,doctor,eq_profile,stream,doctor}.py`,
and in `core/`: `engine.py` (single plugin), `pipeline.py` (chain), `commit.py`
(chain + delay compensation), `mixer.py`, `edl.py`, `voiceover.py`, `latency.py`.

## CLI surface — 26 subcommands

Registered in `cli/app.py:build_parser()`; each module exposes `add_parser(subparsers)`
and `run(args)` (or a per-action `run_*`). Root-level flags (`--json`, `--plain`,
`--verbose`, `--version`) must precede the subcommand — argparse does not accept them
after it.

**Install and inspect**

| Command | Does |
|---|---|
| `scan` | Walk the search paths and refresh the plugin cache |
| `list` | List registered plugins (`--format`, `--vendor`) |
| `info <plugin>` | Metadata + parameter list |
| `dump [plugin]` | Parameters as JSON / JSONL (`--all`, `--filter`, `--save-preset`) |
| `doctor -p <plugin>` | PluginDoctor-style measurements: `--mode linear\|thd\|imd\|sweep\|dynamics\|attack-release\|waveshaper\|performance\|all`, `--compare`, `--clap` |
| `eq-profile -p <plugin>` | EQ profiling: frequency response, phase, group delay, nonlinearity, bypass delta |
| `changelog` | Parse `CHANGELOG.md` (`--since`) |
| `schemas {list,show}` | Publish the JSONSchemas for `--json` output |

**Process audio**

| Command | Does |
|---|---|
| `process <input>` | One plugin over a file or directory (`--passes` for adaptive multipass, `--workers`, `--dry-run`) |
| `chain <input>` | Several plugins in sequence (`--steps "dehum:freq=60,declick"`) |
| `commit <input>` | Chain + delay compensation, written back |
| `preset {save,load,list,delete}` | JSON presets under `~/.audioman/presets/` |

**Mix and edit**

| Command | Does |
|---|---|
| `bounce` | Sum tracks to one stereo file (`--gain`, `--pan`, `--session`) |
| `mixdown` | Tracks through a master chain, with session files |
| `edl {init,add,list,undo,redo,render,status,clear}` | Non-destructive edit list; ops accumulate, `render` applies them |
| `master {prep,qc,verify,list-profiles}` | Delivery workflow against a target loudness profile |
| `fx <input> <effect>` | Built-in DSP: `fade-in`, `fade-out`, `pad`, `remove-dc`, `trim`, `cut-region`, `splice`, `trim-silence`, `normalize`, `gate`, `gain` |
| `fader-test <dir>` | Qt multitrack mixer GUI; exports per-track gains as ground truth |
| `fader-compare <gt>` | Automix recommendation vs. that ground truth |

**Analyze**

| Command | Does |
|---|---|
| `analyze <input>` | RMS, spectral centroid/entropy, silence regions, optional ASCII waveform and spectrum |
| `observe <input>` | Findings across all four categories (`--min-severity`, batch) |
| `screen <input>` | Aesthetic screening: clicks, hum, mouth clicks, sibilance, breath, noise |
| `visualize <input>` | Vamp plugin or built-in analysis → Sonic Visualiser `.svl` (also `--png`) |
| `stream {bench,triage,compare,play}` | DAW reproduction: per-block CPU load, block-boundary click triage, block-size null test, and playback through a real device |

**Voice and video**

| Command | Does |
|---|---|
| `vo {analyze,process}` | VAD → denoise → per-utterance LUFS levelling |
| `obs {probe,dry-run}` | OBS multitrack video: track topology (`probe`), then per-track voice/music classification and a treatment plan (`dry-run`). See `docs/obs-workflow.md` |

## JSON contract

`src/audioman/schemas/*.json` are the published JSON Schemas (draft 2020-12). Each one
carries an `audioman://schema/<name>.v1.json` `$id`, and the `--json` payload of the
matching command carries that URI in `$schema`. Every envelope also has
`audioman_version` and `command`, so a consumer can pin behaviour without parsing
output. Findings-shaped payloads (`observe`, `stream triage`) additionally nest
`findings[]` built from `core/findings.py`, whose `Code` values are append-only:
existing codes are never renamed or removed.

`scripts/check_schema_uris.py` and `tests/unit/test_json_contract.py` enforce both
directions — no schema URI referenced from `src/` may be dangling, and no emitted
envelope may reference an unpublished schema.

## Invariants

1. **One import direction.** `cli/` → `core/` / `plugins/` / `config/`. No `core/` module
   imports `cli/`, and `plugins/` imports neither `core/` nor `cli/`.
2. **Plugins are VST3/AU bundles only.** Discovery is a filesystem glob plus an
   `Info.plist` parse; nothing executes plugin code during `scan`.
3. **`process(audio, sample_rate, reset)` shape.** `audio` is `(channels, samples)`
   `float32`. `reset=False` keeps plugin state across blocks; the streaming layer relies
   on it to reproduce DAW behaviour, the offline path passes the whole buffer at once.
4. **Registry is a process-wide singleton** (`get_registry()`), and its cache file is
   the only persisted plugin state. Concurrent processes can race on
   `~/.audioman/cache/plugins.json`; a torn write is handled by the loader falling back
   to a rescan.
5. **`cxt bs` must report 0 criticals in `src/`.** The sanctioned suppression is a
   category-scoped `# cxt-ignore: <category>` on the offending line with a reason.
6. **No file over 1500 code lines** (`cxt loc --no-blank --no-comments`, Art. III).
   Measured 2026-09-28: largest file is `core/plugin_analysis.py` at 854.
7. **GPU code is optional and unwired.** `gpu/__init__.py:is_available()` imports torch
   lazily, so importing `audioman.gpu` stays cheap. Neither `core/gpu_spectral.py` nor
   `gpu/deverb_unet.py` is reachable from a CLI path or from another module today — they
   are staged for a GPU acceleration phase (`AudiomanSettings.gpu_enabled`, default
   `False`) and are covered by `tests/unit/test_gpu_availability.py` for the capability
   boundary only.

## What will bite you

- **`--json`/`--plain`/`--verbose` are root flags.** `audioman info denoise --json` is a
  hard argparse error. Put them before the subcommand.
- **`--plain` is detected before `parse_args`** (`_early_plain_detect`), because the
  language catalogue is read at import time. Setting `AUDIOMAN_PLAIN=1` in the
  environment is equivalent to passing the flag.
- **`set_parameters` raises `AttributeError`** when a name matches no parameter; a
  value outside the plugin's range or of the wrong type comes back as
  `TypeError`/`ValueError` from pedalboard. `plugin_analysis._set_plugin_parameter`
  returns `False` instead of raising for the sweep paths; the engine path lets it
  propagate.
- **Plugin loading is not free and not thread-safe.** `VST3PluginWrapper.load()` holds a
  module-level lock, and a heavy plugin (a torch-backed de-reverb, say) can take seconds
  and print thousands of lines to stderr while it initializes.
- **Tests must not require an installed plugin.** The suite passes with no VST3 present;
  `tests/unit/test_registry.py` uses synthetic `.vst3` bundles under `tmp_path`.
- **`outputs/` and `testing/` are scratch**, gitignored, and excluded from the LOC gate.
  `testing/` scripts follow §7 of the constitution (dated name, header, no promotion
  without verification).
- **`obs.py` shells out to ffmpeg/ffprobe.** `_ensure_tools()` raises before any work if
  they are missing, and a failing duration probe leaves `duration_sec = 0.0` rather than
  guessing.

## Repository layout

| Path | Holds |
|---|---|
| `src/audioman/cli/` | One module per subcommand, `app.py` (parser + entry point), `output.py` (rich/plain/JSON formatting), `_fader_test_ui.py` (Qt) |
| `src/audioman/core/` | DSP and measurement modules; see the layer table above |
| `src/audioman/plugins/` | `base.py` (Protocol), `vst3.py` (pedalboard wrapper), `parameter.py` (`ParameterInfo`, `PluginMeta`) |
| `src/audioman/config/` | Search paths and settings |
| `src/audioman/gpu/`, `src/audioman/core/gpu_spectral.py` | Optional torch acceleration boundary and helpers |
| `src/audioman/schemas/` | Published JSON Schemas |
| `tests/unit/` | The suite (`pytest tests -q`); `tests/conftest.py` carries the audio fixtures |
| `scripts/` | `check_schema_uris.py`, `gates.py` (local Art. II/III gate echo) |
| `docs/` | `obs-workflow.md` |
| `.github/workflows/` | `constitution.yml` (book.md gate), `python-ci.yml` (tests + coverage report) |
| `presets/`, `outputs/`, `testing/` | Reference presets; scratch output; experimental scripts (the latter two are gitignored) |

Entry points: `audioman` console script → `audioman.cli.app:main`; `python -m audioman`
→ `audioman/__main__.py` → same.
