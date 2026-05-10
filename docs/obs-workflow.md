# OBS multitrack 영상 자동 진단 워크플로우

OBS Studio로 녹화한 영상에서 음성·음악 트랙을 자동 식별하고, 트랙별로 필요한 후처리(디노이즈/디험/디클립/스템 분리/라우드니스 평탄화)를 결정해 주는 dry-run 진단 파이프라인.

## 도입 배경

OBS는 한 영상에 audio stream을 여러 개 인코딩할 수 있다 (마이크, BGM, 시스템 사운드 등을 분리 트랙으로). 그러나:

- "Multitrack Audio" 옵션이 켜져 있지 않으면 **모든 스트림에 같은 마스터 믹스가 복제**되거나 **첫 번째 트랙에만 풀믹스**가 들어간다.
- 음성 위주 트랙에 voice de-noise를 적용하면 좋지만, 음성+음악이 섞인 풀믹스에 같은 처리를 하면 **음악이 손상**된다.
- 인벤토리가 수십 개 되면 어느 파일이 어떤 구조인지 일일이 확인하기 어렵다.

`audioman obs` 서브명령은 이 문제를 자동화한다. 모든 영상의 트랙 토폴로지를 식별하고, 활성 트랙별로 음성/음악/풀믹스/무음을 분류한 뒤, 진단 결과에 맞는 처치 계획(dry-run)을 JSON으로 만든다. 실제 처리는 사람이 검토 후 수동 또는 후속 명령으로 실행한다.

## CLI

### `audioman obs probe`

트랙 토폴로지만 빠르게 식별 (RMS만 측정).

```bash
# 단일 파일
audioman obs probe input.mov

# 디렉터리 (.mov/.mp4/.mkv/.m4v 자동 인식, 같은 stem의 mov/mp4 중복 제거)
audioman obs probe /Volumes/T7/OBS/

# 추출 길이 조정 (기본 15초)
audioman obs probe input.mov --probe-seconds 30

# JSON
audioman --json obs probe /Volumes/T7/OBS/ > topology.json
```

출력 (테이블 모드):

```
file                       topology     streams  active       groups            duration
2025-03-20 10-18-26.mp4    multitrack   6        0,1          [0] | [1]         5152.2s
2025-03-21 11-38-47.mov    single       6        0            [0]               5696.5s
2026-04-30 14-52-59.mov    duplicated   6        0,1,2,3,4,5  [0,1,2,3,4,5]     5.7s
2025-04-18 12-43-31.mov    silent       6        -            -                 5405.9s
```

### 토폴로지 분류

| 값 | 의미 | 처리 전략 |
|----|------|-----------|
| `multitrack` | 활성 트랙들의 RMS가 다름 → 진짜 멀티트랙 | 그룹별로 한 트랙만 분석, 같은 그룹은 결과 미러링 |
| `single` | 활성 트랙이 1개 (나머지는 무음) | 1번 트랙만 분석 |
| `duplicated` | 활성 트랙들의 RMS가 동일 → 같은 신호 복제 | 첫 활성 트랙만 분석, 나머진 결과 복사 |
| `silent` | 모든 트랙 무음 | skip |

`unique_signal_groups`는 같은 RMS를 보인 트랙끼리의 인덱스 그룹. 예를 들어 `[[0,1], [2,3]]`은 트랙 0·1이 같은 신호(스테레오 페어이거나 2채널 복제), 2·3도 마찬가지라는 뜻. dry-run 시 트랙 0과 2만 분석하고 1·3은 같은 처치를 받도록 자동 미러링한다.

### `audioman obs dry-run`

활성 트랙별로 분류·진단·처치 계획을 만든다 (실제 처리 없음).

```bash
# 60초 분석 (기본), 영상 중간부에서 추출
audioman obs dry-run input.mov --seconds 60

# 분석 시작 시점 명시
audioman obs dry-run input.mov --seconds 30 --start 120

# 디렉터리 일괄 + JSON 리포트 저장
audioman obs dry-run /Volumes/T7/OBS/ --seconds 60 --out-dir reports/

# 머신 리더블 출력
audioman --json obs dry-run input.mov > diagnosis.json
```

출력 (테이블 모드):

```
file                       topology     track          kind     actions                          issues
2026-05-04 15-19-04.mov    multitrack   track 0 (=>1)  fullmix  stem_separate,denoise,declick    warn=0 crit=0
2026-05-04 15-19-04.mov    multitrack   track 2 (=>3)  music    dc_removal,declick               warn=1 crit=0
```

`(=>1)`은 트랙 0의 처치를 트랙 1에도 적용한다는 미러링 표기.

## 트랙 분류 (`classify_track`)

VAD speech 비율 + 스펙트럼 대역 분포 + hf slope를 결합한 휴리스틱. 4종류 + silent.

| kind | 판정 조건 (대략) | 특징 |
|------|------------------|------|
| `voice` | `speech_ratio > 0.4` AND `sub < 10%` AND `presence > 1%` | 마이크 음성 클린 트랙 |
| `music` | `speech_ratio < 0.05` AND `sub > 15%` | BGM, 무대음악 등 |
| `fullmix` | `speech_ratio > 0.2` AND `sub > 15%` | 음성+음악 마스터 믹스 |
| `silent` | rms < 1e-4 | 처리 불필요 |

`confidence` (0~1)와 함께 반환된다. 분석 구간이 짧거나 무음 구간을 잡으면 fullmix로 보수적으로 떨어지는 경향이 있어, `--seconds`를 넉넉히 (60~120초) 주는 것이 정확도에 도움이 된다.

## 처치 룰 엔진 (`recommend_treatment`)

진단 결과(`spectrum_diagnostics` + 핵심 QC) → 처치 계획 변환. 각 처치는 `action`, `plugin_short` (registry 단축명), `params`, `rationale`, `severity`(info/warn/critical)를 가진다.

| Action | Plugin (registry short) | 트리거 |
|--------|-------------------------|--------|
| `dehum` | `de-hum` | 50/60/120Hz 중 하나 이상에서 hum SNR 검출 |
| `declip` | `de-clip` | 클리핑 샘플 1개 이상 (>100이면 critical) |
| `dc_removal` | (DSP 내장 HPF) | DC offset > 0.001 |
| `denoise` | `voice-de-noise` | voice 또는 fullmix(stem 분리 후) |
| `leveling` | (`core/loudness.level_utterances`) | voice 트랙의 발화 단위 LUFS 평탄화 |
| `stem_separate` | (Demucs 또는 RX Music Rebalance) | fullmix 트랙: vocals 분리 후 vocals에만 denoise |
| `declick` | `de-click` | `qc.detect_clicks` 검출 (>5개면 warn) |
| `phase_warning` | (없음) | 스테레오 negative correlation > 20% (모노 호환 위험) |
| `channel_balance` | (게인 보정) | L/R 불균형 > 1.5 dB |
| `loudness_check` | (없음) | music 트랙이 LUFS > -10 (헤드룸 부족) |
| `skip` | — | silent 트랙 |

처치는 `severity`별로 정렬·필터해서 자동화 우선순위를 정하기 좋다.

## JSON 리포트 구조

`--out-dir`로 저장하면 영상 stem별 JSON 1개씩 생성된다:

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
  "notes": ["topology=multitrack, active=[0, 1, 2, 3]", "동일 신호 그룹 발견: [[0,1], [2,3]]"]
}
```

## 권장 후속 처리 (수동/스크립트)

dry-run JSON을 받아 트랙별로 다음 흐름:

1. `mirrors` 목록을 활용해 같은 신호 그룹은 한 번만 처리하고 결과를 ffmpeg로 동일 파일에 매핑.
2. `severity=critical` 처치(주로 declip)를 먼저 적용. RX 10 De-clip를 `audioman process` 또는 별도 스크립트로 호출.
3. `kind=fullmix` 트랙은:
   - **Demucs** (`htdemucs`, MPS 디바이스) 또는 **RX 10 Music Rebalance**로 vocals/other 분리.
   - vocals stem에만 voice-de-noise 적용.
   - 처리된 vocals + 원본 other를 합산해 다시 트랙으로.
4. `kind=voice` 트랙은 `audioman vo process` 그대로 호출 가능 (VAD → denoise → utterance LUFS leveling).
5. `kind=music` 트랙은 EQ/loudness만 점검.
6. ffmpeg `-map`으로 처리된 트랙들을 원본 영상에 remux.

## 자료 검증

51개 OBS 파일(`/Volumes/T7/OBS/`)에 대한 dry-run 결과:

- 토폴로지 분포: multitrack 7 / single 28 / duplicated 13 / silent 3
- 트랙 분류 분포: fullmix 39 / voice 8 / music 5 / silent 3
- 처치 빈도: stem_separate 39, denoise 47, declick 44, declip 10, dehum 8
- Critical: 2건 (클리핑 샘플 53,120 / 1,085)

즉 자료의 80%가 stem 분리 후에야 안전한 voice de-noise가 가능한 풀믹스 구조라는 점이 dry-run으로 명확해졌다.

## 모듈 / 함수 참조

| 함수 | 역할 |
|------|------|
| `core.obs.probe_topology(video)` | ffprobe + 트랙별 RMS로 토폴로지 분류 |
| `core.obs.classify_track(audio, sr)` | VAD + 스펙트럼으로 voice/music/fullmix/silent 판정 |
| `core.obs.diagnose_track(audio, sr, cls)` | spectrum_diagnostics + qc 측정을 통합 |
| `core.obs.recommend_treatment(diag)` | 진단 → 처치 계획 룰 엔진 |
| `core.obs.dry_run_video(video, ...)` | 영상 1개에 대한 위 단계 통합 실행 |

CLI: `cli/obs.py` (`probe`, `dry-run`).
테스트: `tests/unit/test_obs.py` (15 테스트).

## 의존성

- `ffmpeg`, `ffprobe` — PATH에 있어야 함
- 기존 audioman 코어: `core.{analysis, qc, vad, audio_file, loudness, registry}`
- 처치 실행 단계(이 워크플로우 외부): RX 10 VST3 (`voice-de-noise`, `de-hum`, `de-click`, `de-clip`, `music-rebalance`), Demucs (4.x, torch MPS 권장)
