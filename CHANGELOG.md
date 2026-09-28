# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added — LLM-native Phase A
- **--plain global output mode**: turns off all rich color/markup/i18n and prints English ASCII. Also enabled by the `AUDIOMAN_PLAIN=1` environment variable. `--help`/`print_table` fall back to grep/awk-friendly text. Direct response to LLM agent feedback #1.
- **Finding schema** (`audioman.core.findings`): unified fault representation across 4 categories — signal / spectral / plugin / container. Fields: `code` (stable enum), `severity` (info/warn/critical), `where` (file/sample/sec/freq), `measurement`, `hint`, `fix_hint`. Publishes a JSONSchema (`audioman://schema/finding.v1.json`).
- **`audioman observe`**: new first-class command — fault detectors for the `signal+spectral` categories (clipping, DC offset, channel imbalance, leading/trailing/inner silence, mains hum, HF noise floor) emit into a unified `finding[]` array. Supports `--category`, `--severity`, `--recursive`. The JSON envelope is always populated with `duration_sec`, `total_samples`, `sample_rate`, `channels` (response to feedback #3).
- **`audioman changelog`**: CHANGELOG.md parser. `--since X.Y.Z` filter, `--json` envelope. Response to feedback #5.
- **`audioman schemas list|show`**: exposes published JSONSchemas. Lets an LLM agent know the shape of `audioman --json` output before calling it.
- **analyze --json metadata enrichment**: added `$schema`, `audioman_version`, `duration_sec`, `total_samples`, `findings[]` fields. The existing `duration` and `frames` fields are kept for compatibility.
- **New detector module** (`audioman.core.detectors`): `detect_clipping`, `detect_dc_offset`, `detect_channel_imbalance`, `silence_to_findings`, `spectrum_to_findings`. Takes the existing `core/analysis.py` output as-is and adapts it into Findings.

### Added — DAW real-time streaming reproduction / plugin benchmarking & debugging
- **`audioman stream`**: a first-class command that reproduces the fixed-block callback processing of a real DAW (Ableton, etc.) to triage plugin clicks/dropouts. Four subcommands:
  - `bench`: measures real-time CPU load per block size (64/128/256/512/1024) — per-block processing time vs. the real-time deadline ratio (RT factor p50/p99/max), xrun count, estimated concurrent track count.
  - `triage`: detects clicks/discontinuities in block-streamed output and emits them as `finding[]`. Distinguishes "streaming state discontinuity (= DAW click)" from "source content click" by whether the click aligns with a block boundary. Includes a null test against the offline render.
  - `compare`: null-test compares outputs from several block sizes against each other and against offline — detects block-size-dependent bugs.
  - `play`: plays the plugin-through signal in real time via sounddevice + counts PortAudio underflows (real xruns).
- **`audioman.core.streaming`**: deterministic block-wise processing engine. `render_offline` (whole-buffer ground truth) / `render_streamed` (contiguous blocks, `reset_per_block` and `reset_first` options). Verified against pedalboard measurements: contiguous calls with `reset=False` match the offline render bit-for-bit (-600dB), while resetting on every block produces boundary clicks (-7.5dB).
- **`audioman.core.discontinuity`**: click triage detectors. `detect_discontinuities` (MAD-robust sample-diff spike + block-boundary alignment classification), `detect_nonfinite` (NaN/Inf), `null_test` (difference vs. offline after PDC compensation). Reuses the existing `Finding`/`Code.CLICK_DENSITY`/`SAMPLE_DROPOUT`/`NONFINITE_SAMPLES` schema.
- **`audioman.core.rt_bench`**: aggregates `BlockTiming` → `RTBenchReport`. Concurrent track estimate based on worst case (p99/max), warm-up blocks excluded.
- **`VST3PluginWrapper.process(reset=)`**: added a reset argument — `reset=False` keeps internal state (filter history/lookahead) continuous across block streaming. Defaults to True, so existing offline behavior stays backward-compatible.

### Removed
- **Complete removal of the i18n infrastructure**: removed `src/audioman/i18n.py` and the Korean catalog. All 282 `_("...")` calls were converted to plain English string literals. `AUDIOMAN_LANG` is no longer supported. CLI output language is always English (`--plain` is now reduced to turning ANSI/Rich off).

### Changed
- **Session file path/subtype validation**: `tracks[].path` and `output` in `core.session` are now resolved relative to the session directory and **must stay inside it**. `../` escapes and absolute paths outside the session directory are rejected with `SessionPathError` (a `ValueError` subclass) — silent path rewriting has been removed. Absolute paths inside the session directory still work, and `format`/`subtype` are restricted to `ALLOWED_SUBTYPES` (`PCM_16`/`PCM_24`/`PCM_32`/`FLOAT`/`DOUBLE`) with case normalization (`pcm_16` → `PCM_16`). CLI impact: `audioman bounce`/`mixdown --session <file>` no longer proceed on escaping paths or an invalid subtype — they exit 1 with a clear error.

### Fixed
- **`VST3PluginWrapper.set_parameters`**: values outside the plugin-reported min/max and NaN/Inf are no longer silently clamped — they are rejected with a `ValueError` naming the parameter and its bounds (the plugin is left untouched). `process()` likewise rejects empty blocks and NaN/Inf input with a `ValueError` before the plugin is loaded.
- **`audioman observe` failure handling**: inputs that do not exist or cannot be decoded used to die with a traceback; they now use `print_error` + exit 1, and in batch mode per-file failures are aggregated so the remaining files are still processed, then exits 1 with `Batch complete: N succeeded, M failed / T total` (previously the whole batch aborted).
- **`audioman fader-compare` ground truth validation**: broken JSON, a non-object top level, a non-string `source_dir`, missing/unmapped `gains`, and non-numeric gain values are each handled with a clear error and exit 1 instead of a traceback (`JSONDecodeError`/`TypeError`/`ZeroDivisionError`).
- **`--plain` markup leak (AUD-1853)**: `audioman --plain visualize` printed `[dim]...[/dim]` tags literally; it now goes through the `print_info`/`print_markup` path, which strips only the tags and keeps the text. The `--plain` console has `markup=False`, so a string containing tags must never be passed directly to `console.print`.
- **Loss of bracketed tokens in dry-run plans (AUD-1853)**: the `[dry-run]` and `[plugin]` tokens in `process --dry-run`/`chain --dry-run` were interpreted as rich markup and vanished entirely (plugin names disappeared from the plan, leaving only `→`). Plan lines are now printed with the new `print_literal` (`markup=False`, `soft_wrap=True`), so the original text is visible in both modes.
- **Over-eager `_strip_markup` (AUD-1853)**: the regex also removed bracketed groups that are not rich tags, so `info`'s parameter range `[0, 1]` and the progress indicator `[1/2]` disappeared in plain mode. It now removes only the forms rich actually interprets as tags (styles/`link=`/`@handler`) and preserves all other bracketed text.
- **`audioman visualize --open` cross-platform (AUD-1853)**: it only called `open -a 'Sonic Visualiser'` (macOS-only) and failed on Linux. darwin now uses `open -a` and linux uses `xdg-open`, and when no launcher exists or the platform is unsupported it prints a guidance message instead of a traceback.

### Tests
- `tests/unit/test_plain_mode.py` (3), `test_findings.py` (18), `test_observe.py` (5), `test_changelog_cmd.py` (5) — 31 new tests.
- `tests/unit/cli_commands/test_plain_tag_leaks.py` (12) — AUD-1853: verifies command by command that rich tag text does not leak into `--plain` output (while bracket-based evidence such as `[dry-run]`/`[1/2]` is preserved) and that the `[0, 1]` parameter range survives.
- `tests/unit/test_streaming.py` (14) — streaming null test, block-boundary click detection, RT factor monotonicity/block-size dependence. Uses only pedalboard built-ins (no VST3 required).
- `tests/unit/test_session_security.py` (28), `test_vst3_validation.py` (42) — session path-escape/subtype guards and VST3 parameter/block validation.

## [0.2.0] - 2026-05-10

### Added
- **obs**: OBS Studio multitrack video auto-diagnosis commands (`audioman obs probe`, `audioman obs dry-run`)
  - Track topology classification: `multitrack` / `single` / `duplicated` / `silent`
  - Active-track classification: `voice` / `music` / `fullmix` / `silent` (VAD speech ratio + spectral bands + hf slope)
  - Treatment rule engine: inspects hum / clipping / DC offset / clicks / phase / channel imbalance, then produces a treatment plan JSON with RX plugin short names
  - Automatic detection of identical-RMS signal groups + mirror map (mirrors track 0's analysis result onto track 1)
  - Directory batch mode, `--out-dir` saves a per-video JSON report
  - dry-run only — actual processing is left to the user or a follow-up command (a safe review step)
  - Core module: `audioman.core.obs` (probe_topology, classify_track, diagnose_track, recommend_treatment, dry_run_video)
  - 17 unit tests (`tests/unit/test_obs.py`)
  - Workflow document: `docs/obs-workflow.md`
- **eq-profile**: EQ plugin profiling command — frequency response / phase / group delay / nonlinearity measurement. Supports `--mode {response,sweep,nonlinear,all}`, `--sweep-param NAME=v1,v2,...`, `--levels`, `--save-npy`. Core module: `audioman.core.plugin_analysis`.
- **bounce**: bounces multiple tracks into a single stereo file. Per-track `--gain`/`--pan`, `|`-separated `--chain`, YAML/JSON `--session`. Core module: `audioman.core.mixer`.
- **commit**: destructively applies a plugin chain to audio + automatic delay compensation. `--no-compensation`, `--no-tail-trim`, `--dry-run` (latency measurement only). Core module: `audioman.core.commit`.
- **mixdown**: track bounce + master bus chain processing. `--master` chain, automatic delay compensation, `--automix` spectrum-based automatic track gain balancing (`--target` profile).
- **edl**: non-destructive editing (EDL) workflow — `init` / `add` / `list` / `undo` / `redo` / `render` / `status` / `clear`. Core module: `audioman.core.edl`.
- **master**: mastering delivery workflow — `prep` (DC removal, padding, fades, loudness normalization), `qc` (PASS/WARN/FAIL report per target profile), `verify`, `list-profiles`. Core module: `audioman.core.qc`.
- **fader-test**: plays multitrack stems through a PyQt mixer GUI to set per-track gain balance and exports it as ground truth JSON. Core module: `audioman.core.multitrack_player`.
- **fader-compare**: compares `fader-test` ground truth against automix recommended gains. Supports `--target` / `--reference`.
- **vo**: voiceover workflow (`analyze`, `process`) — VAD → denoise → per-utterance LUFS leveling. Core module: `audioman.core.voiceover`.
- **screen**: screens for aesthetic issues such as clicks, hum, mouth clicks, sibilance, breaths, background noise, and RF noise (`--issues`, `--backend {auto,essentia,fallback}`). Core module: `audioman.core.aesthetic`.
- **Core modules**: `audioman.core.automix` (hierarchical gain staging, K-20 calibration), `audioman.core.loudness` (ITU-R BS.1770-4 LUFS / True Peak / LRA + loudness normalization), `audioman.core.session` (YAML/JSON multitrack session loader), `audioman.core.test_signal` (test signal generation for plugin analysis — impulse/sine/two-tone/noise/sweep/multitone/log-sweep deconvolution).

### Changed
- **obs probe**: default behavior changed from looking at only the first 15 seconds of a track to **scanning the whole video**.
  - Fixes tracks that emit signal only sporadically, such as OBS desktop audio, being misclassified as silent because the leading section was quiet.
  - `probe_seconds=None` (the default) means the whole video; passing an explicit number keeps the old behavior (first N seconds only).
  - CLI: can be stated explicitly, e.g. `audioman obs probe --probe-seconds 15`.
  - Regression tests: `test_probe_topology_full_scan_catches_late_signal`, `test_probe_topology_default_is_full_scan`

## [0.1.0] - 2026-03-26

### Added
- **i18n**: locale-based multilingual support (`AUDIOMAN_LANG` environment variable, automatic system locale detection)
  - English by default, Korean catalog included, extensible structure
- **doctor**: PluginDoctor-style plugin analysis engine
  - Analysis modes: linear, thd, imd, sweep, dynamics, attack-release, waveshaper, performance
  - A/B comparison (`--compare`), M/S mode (`--mid-side`)
  - CLAP embedding profiling (`--clap`, `--clap-sweep`, `--clap-output`)
  - waveshaper v2: multiple amplitude levels + averaging over several periods + 256-point resampling
  - `--legacy-waveshaper`, `--ws-levels`, `--ws-points` options
- **batch**: `--workers N` parallel processing (process, chain)
- **stream**: automatic streaming processing for large files (>500MB)
- **ux**: Rich progress bar + ETA (batch processing)
- **visualize**: Sonic Visualiser SVL export (Vamp plugins + built-in analysis)
- **analyze**: audio analysis (RMS, spectral entropy, silence detection, ASCII waveform)
- **fx**: built-in DSP effects (normalize, gate, trim, fade, gain)
- Core CLI: scan, list, info, process, chain, preset, dump
- 35 unit tests (audio_file, dsp, analysis, preset_manager)

### Fixed
- VST3 plugin subdirectory scanning (`**/*.vst3` glob)
- CLAP profiling performance optimization (plugin instance reuse)
