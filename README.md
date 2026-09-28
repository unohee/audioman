# audioman

Cross-platform CLI wrapper for VST3/AU audio plugins. Control commercial audio software like iZotope RX from the command line.

Built for AI agents and automated audio pipelines — the root-level `--json` flag gives machine-readable output for every subcommand.

## Install

```bash
# requires uv (https://docs.astral.sh/uv/)
git clone https://github.com/unohee/audioman.git
cd audioman
uv sync

# Install as global CLI tool
uv tool install -e .
audioman --help
```

## Quick Start

```bash
# Scan system for available plugins
audioman scan

# List registered plugins
audioman list

# Show plugin parameters
audioman info denoise

# Process a single file
audioman process input.wav --plugin denoise --param noise_reduction_db=15 -o output.wav

# Multi-pass adaptive denoising (RX Spectral De-noise)
audioman process input.wav -p denoise \
  --param adaptive_learning=true \
  --param noise_reduction_db=12 \
  --passes 2 -o denoised.wav

# Chain multiple plugins
audioman chain input.wav --steps "dehum:notch_frequency=60,declick,denoise" -o cleaned.wav

# Batch process a directory
audioman process ./input_dir/ -p dereverb -o ./output_dir/ --suffix _dry
audioman process ./input_dir/ -p dereverb -o ./output_dir/ -r  # recursive
```

## Commands

The 26 subcommands mirror `audioman --help`:

| Command | Description |
|---------|-------------|
| `scan` | Scan system for VST3/AU plugins |
| `list` | List registered plugins |
| `info <plugin>` | Plugin details + parameter list |
| `process <input>` | Process audio with a single plugin |
| `chain <input>` | Process audio through multiple plugins sequentially |
| `preset` | Preset management (save/load/list/delete) |
| `dump [plugin]` | Dump plugin parameter state to JSON/JSONL (always machine-readable; `--json` is implied) |
| `analyze <input>` | Audio analysis (RMS, spectral entropy, silence detection, etc.) |
| `fx <input>` | Built-in DSP effects (fade, trim, cut, splice, normalize, gate, gain) |
| `visualize <input>` | Vamp plugin or built-in analysis -> Sonic Visualiser SVL file |
| `doctor -p <plugin>` | Plugin analysis — frequency response, THD, dynamics, waveshaper, performance |
| `eq-profile -p <plugin>` | EQ plugin profiling — frequency response, phase, group delay, nonlinearity |
| `bounce` | Bounce multiple tracks into a single stereo file |
| `commit <input>` | Commit plugin chain to audio with auto delay compensation |
| `mixdown` | Mix tracks with master chain processing |
| `edl` | Non-destructive edit workflow (EDL) |
| `master` | Mastering delivery workflow (prep / qc / verify) |
| `fader-test <input>` | Open a multitrack mixer GUI to set per-track gain balance (export as ground truth JSON) |
| `fader-compare <gt>` | Compare automix recommendations against a fader-test ground truth |
| `vo {analyze,process}` | Voiceover workflow (VAD + denoise + per-utterance LUFS leveling) |
| `screen <input>` | Screen audio for aesthetic issues such as clicks, hum, breaths, sibilance, and noise |
| `obs {probe,dry-run}` | OBS multitrack video auto-diagnosis (dry-run) — track topology + voice/music classification + treatment plan |
| `observe <input>` | Observe audio faults across categories (signal, spectral, plugin, container) |
| `changelog` | Show audioman changelog (LLM-friendly, parses CHANGELOG.md) |
| `schemas {list,show}` | Show audioman JSONSchemas (machine-readable contract for `--json` output) |
| `stream {bench,triage,compare,play}` | Reproduce DAW real-time block processing — benchmark & triage plugin clicks/dropouts |

Global flags (`--json`, `--plain`, `--verbose`, `--version`) are defined at the root, so they must be placed **before** the subcommand (see [JSON Output](#json-output)).

## Plugin Click / Dropout Triage (DAW streaming)

audioman reproduces and automatically diagnoses the clicks you hear in a real DAW (Ableton, etc.). A DAW processes audio in fixed blocks (128/256/512 samples) via callbacks and keeps plugin internal state continuous across blocks — `stream` reproduces that environment and exposes what differs from an offline bounce.

```bash
# Real-time CPU load per block size (RT factor, xrun, concurrent track estimate)
audioman stream bench mix.wav -p reverb --blocks 64,128,256,512,1024

# Click/discontinuity triage — block-boundary aligned = streaming bug, unaligned = source click
audioman --json stream triage mix.wav -p denoise --block-size 512

# Simulate a misbehaving host (reset on every block) — forces clicks
audioman stream triage mix.wav -p denoise --reset-per-block

# Block-size-dependent bug: null test whether output differs per block size
audioman stream compare mix.wav -p delay --blocks 128,256,512

# Play through a real audio device + count PortAudio underflows (real xruns)
audioman stream play mix.wav -p reverb --block-size 256

# Test without VST3: use pedalboard built-in effects via the builtin: prefix
audioman stream triage sine -p builtin:reverb --reset-per-block
```

## Batch Processing

Any file argument can be a directory for batch mode:

```bash
# Process all files in a directory
audioman process ./recordings/ -p denoise -o ./cleaned/

# Recursive (include subdirectories)
audioman process ./recordings/ -p denoise -o ./cleaned/ -r

# Same directory with suffix
audioman process ./recordings/ -p dereverb --param output_reverb_only=true \
  -o ./recordings/ --suffix _deverb
```

## Plugin Parameter Dump

Dump default parameters or full catalog as JSONL:

```bash
# Single plugin state
audioman dump denoise

# With parameter overrides
audioman dump dehum --param notch_frequency=50

# Dump ALL plugins as JSONL
audioman dump --all -o all_plugins.jsonl

# Filter by keyword
audioman dump --all --filter "rx 10" -o rx10_defaults.jsonl
```

## JSON Output

`--json`, `--plain`, `--verbose`, and `--version` are **root-level global flags**. They are parsed by the top-level parser, so they must appear **before** the subcommand. Placing them after the subcommand fails with `error: unrecognized arguments`:

```bash
# Correct — global flags first
audioman --json info denoise
audioman --json process input.wav -p denoise -o out.wav

# Wrong — rejected by argparse
audioman info denoise --json   # error: unrecognized arguments: --json

# Batch mode outputs JSONL (one JSON object per line)
audioman --json process ./dir/ -p denoise -o ./out/
```

`dump` is the one exception to the "flag is required" rule: it is always machine-readable (JSON for a single plugin, JSONL for `--all`), so `--json` is implied and accepted only for uniformity. Every other command needs the root-level flag to switch output modes.

## Presets

```bash
# Save current parameters as a preset
audioman preset save my_denoise --plugin denoise \
  --param noise_reduction_db=20 --param adaptive_learning=true

# List presets
audioman preset list

# Use a preset when dumping plugin state
audioman dump denoise --preset my_denoise

# Dump plugin state and save as preset in one step
audioman dump denoise --param noise_reduction_db=25 --save-preset aggressive_denoise
```

## Audio Analysis

```bash
# Full analysis (RMS, spectral centroid, silence detection)
audioman analyze input.wav

# With ASCII waveform visualization
audioman analyze input.wav -w

# Frame-level metrics (global --json goes before the subcommand)
audioman --json analyze input.wav --frames
```

## Built-in DSP Effects

```bash
# Normalize to -1dB peak
audioman fx input.wav normalize -o output.wav

# Noise gate
audioman fx input.wav gate --threshold -40 -o output.wav

# Trim silence from start and end
audioman fx input.wav trim-silence -o output.wav

# Fade in/out
audioman fx input.wav fade-in --duration 0.5 -o output.wav
```

## Sonic Visualiser Integration

Export analysis data as `.svl` files that Sonic Visualiser can open directly.

```bash
# Built-in spectrogram → SVL
audioman visualize input.wav -b spectrogram -o spec.svl

# Built-in spectral centroid → SVL
audioman visualize input.wav -b spectral-centroid -o centroid.svl

# Vamp plugin analysis → SVL (requires vamp package + plugins)
audioman visualize input.wav -p vamp-example-plugins:powerspectrum -o power.svl
audioman visualize input.wav -p qm-vamp-plugins:qm-chromagram -o chroma.svl

# List installed Vamp plugins
audioman visualize input.wav --list-plugins

# Open in Sonic Visualiser after generation
audioman visualize input.wav -b spectrogram --open
```

Built-in analysis types: `spectrogram`, `spectral-centroid`, `spectral-entropy`, `rms`, `peak`, `zcr`

## OBS multitrack Diagnosis

Automatically diagnoses the audio streams of OBS Studio videos, identifies which tracks are voice/music/fullmix, and produces a treatment plan (dry-run).

```bash
# Track topology only, quickly (multitrack/single/duplicated/silent)
audioman obs probe /Volumes/T7/OBS/

# 60-second analysis + diagnosis + treatment plan JSON (no actual processing)
audioman obs dry-run /Volumes/T7/OBS/ --seconds 60 --out-dir reports/
```

Each active track is classified as `voice` / `music` / `fullmix` / `silent`, and a treatment matching the hum / clipping / DC offset / clicks / phase check results (dehum, declip, voice-de-noise, leveling, stem_separate, etc.) is chosen. Identical signal groups are mirrored automatically and receive the same treatment.

For detailed usage, classification rules, the JSON report structure, and follow-up processing guidance, see [docs/obs-workflow.md](docs/obs-workflow.md).

## Plugin Analysis (Doctor)

PluginDoctor-style measurements for any VST3/AU plugin:

```bash
# Full analysis (frequency response, THD, dynamics, waveshaper, performance)
audioman doctor -p denoise

# Single mode
audioman doctor -p saturn-2 --mode thd --frequency 1000 --level -6

# Waveshaper v2 (multi-level measurement)
audioman doctor -p decapitator --mode waveshaper --ws-levels 7 --ws-points 256

# A/B comparison
audioman doctor -p saturn-2 --compare vsm-3

# CLAP embedding profiling
audioman doctor -p vsm-3 --clap --clap-sweep drive=0,25,50,75,100 \
  --clap-output vsm3_embeddings.npy
```

Modes: `linear`, `thd`, `imd`, `sweep`, `dynamics`, `attack-release`, `waveshaper`, `performance`, `all`

### Vamp Plugin Setup (macOS)

```bash
pip install vamp
brew install vamp-plugin-sdk
mkdir -p ~/Library/Audio/Plug-Ins/Vamp
cp /opt/homebrew/lib/vamp/vamp-example-plugins.* ~/Library/Audio/Plug-Ins/Vamp/
# macOS requires .dylib extension
cp ~/Library/Audio/Plug-Ins/Vamp/vamp-example-plugins.so \
   ~/Library/Audio/Plug-Ins/Vamp/vamp-example-plugins.dylib
```

For QM Vamp Plugins (chromagram, tempo tracking, onset detection), download from [vamp-plugins.org](https://www.vamp-plugins.org/download.html).

## Verified Plugins

Tested with iZotope RX 10 (15 VST3 plugins, all parameters accessible):

| Plugin | Short Name | Aliases | Params |
|--------|-----------|---------|--------|
| Spectral De-noise | `spectral-de-noise` | `denoise` | 27 |
| Voice De-noise | `voice-de-noise` | `voice-denoise` | 14 |
| Guitar De-noise | `guitar-de-noise` | `guitar-denoise` | 13 |
| De-click | `de-click` | `declick` | 6 |
| De-clip | `de-clip` | `declip` | 10 |
| De-crackle | `de-crackle` | `decrackle` | 5 |
| De-ess | `de-ess` | `deess` | 9 |
| De-hum | `de-hum` | `dehum` | 33 |
| De-plosive | `de-plosive` | `deplosive` | 4 |
| De-reverb | `de-reverb` | `dereverb` | 10 |
| Breath Control | `breath-control` | - | 4 |
| Mouth De-click | `mouth-de-click` | `mouth-declick` | 4 |
| Repair Assistant | `repair-assistant` | `repair` | 15 |

Any VST3 or AU plugin installed on the system can be used — not limited to iZotope.

## Requirements

- macOS (primary, VST3 + AU)
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- VST3/AU plugins installed on the system

## Stack

- [pedalboard](https://github.com/spotify/pedalboard) — Spotify's plugin hosting engine
- numpy — DSP, FFT, spectral analysis
- [vamp](https://pypi.org/project/vamp/) — Vamp plugin host (optional, for `visualize` command)
- argparse — CLI
- rich — terminal output
- pydantic-settings — configuration
- soundfile — audio I/O

## License

MIT
