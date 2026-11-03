# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added — `audioman diff A B`
- Aligns B to A by steady-state cross-correlation (when the lag can be established; otherwise compares unaligned and says why), describes both with the texture descriptors, and reports B - A for the whole file, every segment, every ERB band and six named bands (sub, bass, low_mid, mid, presence, air). A delta is **significant** only when it exceeds 2 x (floor_A + floor_B) and a small per-unit minimum, so identical inputs, or a delayed copy, report no significant change.
- `changes` lists the significant whole-file changes, most significant (relative to floor) first; `residual_rel_db` gives the aligned sample-level difference; `--png` draws the per-band change against its floor band and every descriptor change in units of its floor; JSON carries a digest and a report.v2 block (`alignment` gate, `delta.*` measurements with repeat floors) and honours `--detail`.

### Fixed
- Latency/alignment estimation subtracts the mean first: a DC offset correlates with itself at every lag and could outweigh the signal.

### Added — texture descriptors (`analyze --texture`)
- **`audioman.core.texture.describe`** and **`analyze --texture [--segment-sec S]`**: spectral centroid, rolloff, bandwidth, flatness, contrast, flux, HF ratio, approximate sharpness, RMS, crest factor, onset density, stereo width and L/R correlation, for the whole file and per segment, plus a 32-band ERB level profile.
- **Repeat floor** on every descriptor: each is measured again on the signal advanced by half an analysis hop (every frame and segment boundary moves, the content does not); the difference is its `floor` (whole file) and `segment_floor` (largest per-segment difference). Rolloff, which lands on FFT bins, never gets a floor below one bin. A change smaller than the floor is not evidence of a change.
- report.v2 gains the `repeat` floor kind; `analyze --texture` puts the descriptors into the report as `texture.*` measurements with their repeat floors.
- Direction is pinned on synthetic processing (lowpass, saturation, added noise, decorrelated channels, transient rate), each change exceeding three times its floor.

### Added — plugin/host acceptance gates (`audioman uat`)
- **`uat compare`** judges a host's real-time take against its offline render and the dry source (and, optionally, a bypassed take and the render time). Gates: `realtime_present`, `offline_present`, `rt_matches_offline` (residual <= -90 dBFS at lag 0), `no_dropout_blocks`, `no_unprocessed_blocks`, `delay_compensated` (offline render at lag 0 from the source), `processed` (drop with `--transparent`), `bypass_equals_source`, `faster_than_real_time`.
- **`uat smoke`** renders a tone+noise probe through a plugin (VST3/AU or `builtin:`) at each `--rates` value and checks shape, finiteness, an audible second half and exposed parameters; `--check-output` judges a render made by another host.
- **`uat pdc`** measures latency by steady-state cross-correlation (settling period cut, two-sided search, edge-of-window flag) and compares it with the reported latency (+/-2 ms, NCC >= 0.70). A latency the measurement cannot establish fails.
- Results are report.v2 (`audioman://schema/uat.v1.json`); the exit status is 1 when the verdict is FAIL.
- **`stream triage`** reports dropped host buffers: 256-sample blocks that are silent (< -60 dBFS) where the offline render is active (>= -40 dBFS) produce a critical `SAMPLE_DROPOUT` finding.
- Written from general DSP practice; no code or fixtures from other repositories.

### Added — report.v2
- **`report` block** (`audioman://schema/report.v2.json`) on `analyze`, `observe`, `screen`, `master qc` and `doctor` JSON output: `subject`, `measurements` (value, unit, **floor**, `floor_kind`, source), `findings`, `gates` (PASS/WARN/FAIL), `missing_gates` and `verdict`. A floor is the smallest difference in a value that means anything: exact for counts, half the last kept decimal for rounded values, one sample period for durations, one LSB for linear levels of PCM files, null only when declared `unknown`.
- Each command declares the gates it must produce (`no_critical_findings`, `no_fail_events`, one per QC check with phase/imbalance required on stereo, `completed.<mode>` per doctor mode). A declared gate that was not produced is present as **FAIL, reason `missing`** — a check that did not run is never a pass.
- The block is additive: v1 fields are unchanged. Under `--detail` its lists are trimmed (gates never), but it is never moved to a spill file, so it always validates in place.

### Added — token-budgeted JSON
- **`--detail {summary,standard,full}`** and **`--spill-dir`** on `analyze`, `observe`, `screen` and `master qc`. `standard` (default) keeps the printed JSON under 32 KiB, `summary` under 4 KiB; long numeric arrays, long strings and long finding/event/check lists move to JSON spill files (default `<cache dir>/spill`) referenced by absolute path, with length and min/max/mean inline. Lists keep their most severe items first. Nothing is dropped: `detail.spilled` and `detail.truncated` list every file.
- **`digest`**: every payload of those commands starts with a plain-English summary of the full measurement (format, levels, silence, hum, finding/event counts, QC verdict), at most 600 characters.

### Changed
- **Default JSON shape for long arrays**: with the default `--detail standard`, `analyze --frames` per-frame arrays (and any other array over 64 values) are references to spill files instead of inline lists. Pass `--detail full` for the previous inline output.

### Performance
- **`master qc` ~9x faster, `screen` ~4x faster** on a 5-minute stereo 48 kHz file (in-process, measured by `scripts/bench_perf.py` against the previous per-window implementations).
  - Short-term LUFS / LRA: the signal is K-weighted once and window energies come from a cumulative sum (was one pyloudnorm filter pass per 3 s window, ~3000 per 5 minutes). `measure()` shares that pass with integrated loudness. True peak oversamples float32 input in float32.
  - `screen` (sibilance, breath, background noise): one batched, chunked STFT per frame size (`audioman.core.framewise`) instead of an rfft per frame and per band.
  - `master qc` stereo correlation: all 100 ms windows correlated at once.
  - `screen` RF-noise floor: interior bins take their neighbourhood median in one batched call (was one `np.median` per candidate bin, ~12k per file).
  - `scripts/bench_perf.py` compares CPU time, not wall time, so host load from other processes does not move the asserted ratio.
  - Fewer large temporaries: `framewise` reuses per-call work buffers (`out=`) and derives RMS/ZCR from whole-signal cumulative sums; `detect_clicks` builds its rolling-RMS ratio in place; `long_term_spectrum` (hum, `analyze --spectrum`, `obs`) batches its FFTs; `screen` downmixes once instead of once per detector. On the python.org CPython 3.12 macOS build a fresh multi-MB array costs 2-4x the arithmetic done in it, so this is what keeps the speedup there.
- **Behaviour note:** short-term loudness now runs the K-weighting filter continuously across windows, as a short-term meter does (EBU Tech 3341), instead of restarting it per window. Values move by < 0.03 LU on steady material; a window that begins exactly as a signal stops now reports the filter's decaying tail (a very low finite level) instead of `-inf`. Parity is pinned by `tests/unit/test_perf_parity.py`.

### Fixed — VST3 discovery on Linux/Windows
- **`audioman scan` on Linux/Windows**: VST3 bundles without `Contents/Info.plist` (e.g. `<name>.vst3/Contents/x86_64-linux/<name>.so`) were silently skipped (`Scanned 0 plugins`). They now register under the bundle name, with vendor/version from `Contents/Resources/moduleinfo.json` when present. The inner `Contents/x86_64-win/<name>.vst3` binary is not registered a second time, and a legacy single-file Windows `.vst3` registers under its file stem. A directory without `Contents/` is still ignored, and a corrupt macOS `Info.plist` still yields no plugin.

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
