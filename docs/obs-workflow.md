# OBS multitrack video auto-diagnosis workflow

A dry-run diagnosis pipeline that automatically identifies voice and music tracks in videos recorded with OBS Studio and decides on the post-processing each track needs (denoise / dehum / declip / stem separation / loudness leveling).

## Background

OBS can encode several audio streams into a single video (microphone, BGM, system sound, etc. as separate tracks). However:

- Unless the "Multitrack Audio" option is enabled, **the same master mix is duplicated onto every stream** or **the full mix lands only on the first track**.
- Voice de-noise works well on a voice-dominant track, but applying the same processing to a full mix that contains both voice and music **damages the music**.
- Once the inventory reaches dozens of files, checking which file has which structure by hand becomes impractical.

The `audioman obs` subcommands automate this. They identify the track topology of every video, classify each active track as voice/music/fullmix/silent, and then produce a dry-run treatment plan in JSON that matches the diagnosis. Actual processing is executed after human review, either manually or by a follow-up command.

## CLI

### `audioman obs probe`

Quickly identifies track topology only (measures RMS alone).

```bash
# Single file
audioman obs probe input.mov

# Directory (auto-detects .mov/.mp4/.mkv/.m4v, deduplicates mov/mp4 with the same stem)
audioman obs probe /Volumes/T7/OBS/

# Adjust extraction length (15 seconds by default)
audioman obs probe input.mov --probe-seconds 30

# JSON
audioman --json obs probe /Volumes/T7/OBS/ > topology.json
```

Output (table mode):

```
file                       topology     streams  active       groups            duration
2025-03-20 10-18-26.mp4    multitrack   6        0,1          [0] | [1]         5152.2s
2025-03-21 11-38-47.mov    single       6        0            [0]               5696.5s
2026-04-30 14-52-59.mov    duplicated   6        0,1,2,3,4,5  [0,1,2,3,4,5]     5.7s
2025-04-18 12-43-31.mov    silent       6        -            -                 5405.9s
```

### Topology classification

| Value | Meaning | Processing strategy |
|----|------|-----------|
| `multitrack` | Active tracks have different RMS → a genuine multitrack | Analyze one track per group, mirror the result to identical groups |
| `single` | One active track (the rest are silent) | Analyze track 1 only |
| `duplicated` | Active tracks have identical RMS → the same signal duplicated | Analyze the first active track only, copy the result to the rest |
| `silent` | All tracks silent | skip |

`unique_signal_groups` holds the index groups of tracks that show the same RMS. For example, `[[0,1], [2,3]]` means tracks 0 and 1 carry the same signal (a stereo pair or a 2-channel duplicate) and so do 2 and 3. During dry-run only tracks 0 and 2 are analyzed, and tracks 1 and 3 are auto-mirrored to receive the same treatment.

### `audioman obs dry-run`

Builds the classification, diagnosis, and treatment plan for each active track (no actual processing).

```bash
# 60-second analysis (default), extracted from the middle of the video
audioman obs dry-run input.mov --seconds 60

# Explicit analysis start time
audioman obs dry-run input.mov --seconds 30 --start 120

# Directory batch + save JSON reports
audioman obs dry-run /Volumes/T7/OBS/ --seconds 60 --out-dir reports/

# Machine-readable output
audioman --json obs dry-run input.mov > diagnosis.json
```

Output (table mode):

```
file                       topology     track          kind     actions                          issues
2026-05-04 15-19-04.mov    multitrack   track 0 (=>1)  fullmix  stem_separate,denoise,declick    warn=0 crit=0
2026-05-04 15-19-04.mov    multitrack   track 2 (=>3)  music    dc_removal,declick               warn=1 crit=0
```

`(=>1)` means track 0's treatment is also applied to track 1.

## Track classification (`classify_track`)

A heuristic combining VAD speech ratio + spectral band distribution + hf slope. Four kinds plus silent.

| kind | Decision conditions (approximate) | Characteristics |
|------|------------------|------|
| `voice` | `speech_ratio > 0.4` AND `sub < 10%` AND `presence > 1%` | Clean microphone voice track |
| `music` | `speech_ratio < 0.05` AND `sub > 15%` | BGM, live music, etc. |
| `fullmix` | `speech_ratio > 0.2` AND `sub > 15%` | Master mix of voice + music |
| `silent` | rms < 1e-4 | No processing needed |

Returned together with `confidence` (0~1). A short analysis window or one that lands on a silent section tends to fall back conservatively to fullmix, so giving a generous `--seconds` (60~120 seconds) helps accuracy.

## Treatment rule engine (`recommend_treatment`)

Diagnosis results (`spectrum_diagnostics` + core QC) → treatment plan. Each treatment has `action`, `plugin_short` (registry short name), `params`, `rationale`, and `severity` (info/warn/critical).

| Action | Plugin (registry short) | Trigger |
|--------|-------------------------|--------|
| `dehum` | `de-hum` | hum SNR detected at one or more of 50/60/120Hz |
| `declip` | `de-clip` | one or more clipped samples (critical above 100) |
| `dc_removal` | (built-in DSP HPF) | DC offset > 0.001 |
| `denoise` | `voice-de-noise` | voice, or fullmix (after stem separation) |
| `leveling` | (`core/loudness.level_utterances`) | per-utterance LUFS leveling of a voice track |
| `stem_separate` | (Demucs or RX Music Rebalance) | fullmix track: separate vocals, then denoise the vocals only |
| `declick` | `de-click` | detected by `qc.detect_clicks` (warn above 5) |
| `phase_warning` | (none) | stereo negative correlation > 20% (mono compatibility risk) |
| `channel_balance` | (gain correction) | L/R imbalance > 1.5 dB |
| `loudness_check` | (none) | music track at LUFS > -10 (insufficient headroom) |
| `skip` | — | silent track |

Treatments can be sorted and filtered by `severity`, which is handy for deciding automation priority.

## JSON report structure

Saving with `--out-dir` creates one JSON file per video stem:

```json
{
  "video": "/Volumes/T7/OBS/2026-05-04 15-19-04.mov",
  "topology": {
    "topology": "multitrack",
    "n_streams": 6,
    "active_indices": [0, 1, 2, 3],
    "unique_signal_groups": [[0, 1], [2, 3]],
    "sample_rate": 48000,
    "duration_sec": 20.833,
    "tracks": [{"index": 0, "rms": 0.114, "is_silent": false, "is_stereo": true}, ...]
  },
  "tracks": [
    {
      "track_index": 0,
      "analysis_start_sec": 0.0,
      "analysis_seconds": 20.83,
      "classification": {"kind": "fullmix", "confidence": 0.6, "speech_ratio": 0.254, ...},
      "loudness": {"integrated_lufs": -16.13, "true_peak_dbtp": -0.47, ...},
      "spectrum": {"band_energy": [...], "dominant_frequencies": [...], "hum_check": [...], "hf_slope": {...}},
      "clipping": {"n_samples": 0, ...},
      "dc_offset_max": 0.0001,
      "clicks": {"n_clicks": 3, ...},
      "head_tail_silence": {"head_ms": 12.5, "tail_sec": 0.04},
      "phase": {"applicable": true, "min_window_correlation": 0.99, ...},
      "channel_imbalance": {"applicable": true, "imbalance_db": 0.0}
    }
  ],
  "treatments": [
    {
      "track_index": 0,
      "kind": "fullmix",
      "mirrors": [1],
      "plan": [
        {"action": "stem_separate", "plugin": null, "params": {...}, "rationale": "...", "severity": "info"},
        {"action": "denoise", "plugin": "voice-de-noise", "params": {"apply_to": "vocals_stem_only"}, ...}
      ]
    }
  ],
  "notes": ["topology=multitrack, active=[0, 1, 2, 3]", "Identical signal groups found: [[0,1], [2,3]]"]
}
```

## Recommended follow-up processing (manual/scripted)

Taking the dry-run JSON, the per-track flow is:

1. Using the `mirrors` list, process each identical signal group only once and map the result back onto the same files with ffmpeg.
2. Apply `severity=critical` treatments (mostly declip) first. Call RX 10 De-clip through `audioman process` or a separate script.
3. For `kind=fullmix` tracks:
   - Separate vocals/other with **Demucs** (`htdemucs`, MPS device) or **RX 10 Music Rebalance**.
   - Apply voice-de-noise to the vocals stem only.
   - Sum the processed vocals with the original other and write it back as a track.
4. For `kind=voice` tracks, `audioman vo process` can be called as-is (VAD → denoise → utterance LUFS leveling).
5. For `kind=music` tracks, only check EQ/loudness.
6. Remux the processed tracks back into the original video with ffmpeg `-map`.

## Data verification

Dry-run results over 51 OBS files (`/Volumes/T7/OBS/`):

- Topology distribution: multitrack 7 / single 28 / duplicated 13 / silent 3
- Track classification distribution: fullmix 39 / voice 8 / music 5 / silent 3
- Treatment frequency: stem_separate 39, denoise 47, declick 44, declip 10, dehum 8
- Critical: 2 cases (clipped samples 53,120 / 1,085)

In other words, the dry-run made it clear that 80% of the material is fullmix-structured, where safe voice de-noise is only possible after stem separation.

## Module / function reference

| Function | Role |
|------|------|
| `core.obs.probe_topology(video)` | Classifies topology from ffprobe + per-track RMS |
| `core.obs.classify_track(audio, sr)` | Decides voice/music/fullmix/silent from VAD + spectrum |
| `core.obs.diagnose_track(audio, sr, cls)` | Combines spectrum_diagnostics + qc measurements |
| `core.obs.recommend_treatment(diag)` | Diagnosis → treatment plan rule engine |
| `core.obs.dry_run_video(video, ...)` | Runs all of the above for a single video |

CLI: `cli/obs.py` (`probe`, `dry-run`).
Tests: `tests/unit/test_obs.py` (17 tests).

## Dependencies

- `ffmpeg`, `ffprobe` — must be on PATH
- Existing audioman core: `core.{analysis, qc, vad, audio_file, loudness, registry}`
- Treatment execution stage (outside this workflow): RX 10 VST3 (`voice-de-noise`, `de-hum`, `de-click`, `de-clip`, `music-rebalance`), Demucs (4.x, torch MPS recommended)
