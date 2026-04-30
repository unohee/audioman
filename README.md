# audioman

Cross-platform CLI wrapper for VST3/AU audio plugins. Control commercial audio software like iZotope RX from the command line.

Built for AI agents and automated audio pipelines — every command supports `--json` output.

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

| Command | Description |
|---------|-------------|
| `scan` | Discover VST3/AU plugins on the system |
| `list` | List registered plugins with filters |
| `info <plugin>` | Show plugin parameters and ranges |
| `process <input>` | Process audio with a single plugin |
| `chain <input>` | Sequential multi-plugin processing |
| `preset` | Save/load/list/delete parameter presets |
| `dump <plugin>` | Dump plugin parameter state as JSON |
| `analyze <input>` | Audio analysis (RMS, spectral, silence detection) |
| `fx <input>` | Built-in DSP effects (normalize, gate, trim, fade) |
| `visualize <input>` | Export analysis to Sonic Visualiser SVL files |
| `doctor -p <plugin>` | Plugin analysis (freq response, THD, dynamics, waveshaper) |

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

All commands support `--json` for machine-readable output:

```bash
audioman --json info denoise
audioman --json process input.wav -p denoise -o out.wav

# Batch mode outputs JSONL (one JSON object per line)
audioman --json process ./dir/ -p denoise -o ./out/
```

## Presets

```bash
# Save current parameters as a preset
audioman preset save my_denoise --plugin denoise \
  --param noise_reduction_db=20 --param adaptive_learning=true

# List presets
audioman preset list

# Use preset during processing
audioman process input.wav -p denoise --preset my_denoise -o out.wav

# Dump plugin state and save as preset in one step
audioman dump denoise --param noise_reduction_db=25 --save-preset aggressive_denoise
```

## Audio Analysis

```bash
# Full analysis (RMS, spectral centroid, silence detection)
audioman analyze input.wav

# With ASCII waveform visualization
audioman analyze input.wav -w

# Frame-level metrics
audioman analyze input.wav --frames --json
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

## Internationalization (i18n)

CLI help text supports locale-based translation. Default is English; Korean is included.

```bash
# Force language via environment variable
AUDIOMAN_LANG=ko audioman --help   # Korean
AUDIOMAN_LANG=en audioman --help   # English

# Auto-detects from system locale (LC_ALL, LANG)
audioman --help
```

To add a new language, add a catalog dict to `src/audioman/i18n.py`.

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
