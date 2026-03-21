# audioman

Cross-platform CLI wrapper for VST3/AU audio plugins. Control commercial audio software like iZotope RX from the command line.

Built for AI agents and automated audio pipelines — every command supports `--json` output.

## Install

```bash
# requires uv (https://docs.astral.sh/uv/)
git clone https://github.com/unohee/audioman.git
cd audioman
uv sync
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
- argparse — CLI
- rich — terminal output
- pydantic-settings — configuration
- soundfile — audio I/O

## License

MIT
