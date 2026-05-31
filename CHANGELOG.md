# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added — LLM-native Phase A
- **--plain 글로벌 출력 모드**: rich color/markup/i18n을 모두 끄고 영어 ASCII 출력. `AUDIOMAN_PLAIN=1` 환경변수로도 활성화. `--help`/`print_table`이 grep/awk 친화 텍스트로 fallback. LLM agent 후기 #1 직접 대응.
- **Finding 스키마** (`audioman.core.findings`): signal / spectral / plugin / container 4개 카테고리 통합 결함 표현. `code` (안정 enum), `severity` (info/warn/critical), `where` (file/sample/sec/freq), `measurement`, `hint`, `fix_hint` 필드. JSONSchema 발행 (`audioman://schema/finding.v1.json`).
- **`audioman observe`**: 새 1급 명령 — `signal+spectral` 카테고리 fault detector(clipping, DC offset, channel imbalance, leading/trailing/inner silence, mains hum, HF noise floor)를 통합된 `finding[]` 배열로 emit. `--category`, `--severity`, `--recursive` 지원. JSON envelope에 `duration_sec`, `total_samples`, `sample_rate`, `channels` 항상 채워짐 (후기 #3 대응).
- **`audioman changelog`**: CHANGELOG.md 파서. `--since X.Y.Z` 필터, `--json` envelope. 후기 #5 대응.
- **`audioman schemas list|show`**: 발행된 JSONSchema 노출. LLM agent가 `audioman --json` 출력의 모양을 호출 전에 알 수 있다.
- **analyze --json 메타 보강**: `$schema`, `audioman_version`, `duration_sec`, `total_samples`, `findings[]` 필드 추가. 기존 `duration`, `frames` 필드는 호환을 위해 유지.
- **새 detector 모듈** (`audioman.core.detectors`): `detect_clipping`, `detect_dc_offset`, `detect_channel_imbalance`, `silence_to_findings`, `spectrum_to_findings`. 기존 `core/analysis.py` 출력을 그대로 받아 Finding으로 어댑팅.

### Added — DAW 실시간 스트리밍 재현 / 플러그인 벤치마크·디버깅
- **`audioman stream`**: 실제 DAW(Ableton 등)의 고정 블록 콜백 처리를 재현해 플러그인 클릭/드롭아웃을 triage 하는 1급 명령. 4개 서브커맨드:
  - `bench`: 블록 크기(64/128/256/512/1024)별 실시간 CPU 부하 측정 — 블록당 처리시간 vs 실시간 마감(deadline) 비율(RT factor p50/p99/max), xrun 수, 추정 동시 트랙 수.
  - `triage`: 블록 스트리밍 출력에서 클릭/불연속을 검출해 `finding[]`로 emit. 블록 경계 정렬 여부로 "스트리밍 상태 단절(=DAW 클릭)" vs "소스 콘텐츠 클릭"을 구분. offline 렌더 대비 null test 포함.
  - `compare`: 여러 블록 크기 출력을 서로/오프라인과 null test 비교 — block-size 의존 버그 탐지.
  - `play`: sounddevice로 플러그인 통과 신호 실시간 재생 + PortAudio underflow(실 xrun) 카운트.
- **`audioman.core.streaming`**: 블록 단위 결정적 처리 엔진. `render_offline`(whole-buffer ground truth) / `render_streamed`(연속 블록, `reset_per_block`·`reset_first` 옵션). pedalboard 실측 확인: `reset=False` 연속 호출은 오프라인 렌더와 비트 단위 일치(-600dB), 매 블록 reset 시 경계 클릭(-7.5dB).
- **`audioman.core.discontinuity`**: 클릭 triage 디텍터. `detect_discontinuities`(MAD-robust sample-diff spike + 블록 경계 정렬 분류), `detect_nonfinite`(NaN/Inf), `null_test`(PDC 보상 후 offline 대비 차이). 기존 `Finding`/`Code.CLICK_DENSITY`/`SAMPLE_DROPOUT`/`NONFINITE_SAMPLES` 스키마 재사용.
- **`audioman.core.rt_bench`**: `BlockTiming`→`RTBenchReport` 집계. worst-case(p99/max) 기반 동시 트랙 추정, 워밍업 블록 제외.
- **`VST3PluginWrapper.process(reset=)`**: reset 인자 추가 — 블록 스트리밍에서 `reset=False`로 내부 상태(필터 히스토리/lookahead) 연속 유지. 기본값 True로 기존 오프라인 동작 하위호환.

### Removed
- **i18n 인프라 전면 삭제**: `src/audioman/i18n.py` 및 한국어 카탈로그 제거. 282개 `_("...")` 호출을 모두 평문 영어 문자열로 변환. `AUDIOMAN_LANG` 환경변수 미지원. CLI 출력 언어는 항상 영어로 통일 (`--plain` 플래그의 역할은 ANSI/Rich 끄기로 축소).

### Tests
- `tests/unit/test_plain_mode.py` (3), `test_findings.py` (16), `test_observe.py` (5), `test_changelog_cmd.py` (5) — 신규 29개 추가.
- `tests/unit/test_streaming.py` (14) — streaming null test, 블록 경계 클릭 검출, RT factor 단조성/블록 크기 의존성. pedalboard 빌트인만 사용(VST3 불필요).

## [0.2.0] - 2026-05-10

### Added
- **obs**: OBS Studio 멀티트랙 영상 자동 진단 명령 (`audioman obs probe`, `audioman obs dry-run`)
  - 트랙 토폴로지 분류: `multitrack` / `single` / `duplicated` / `silent`
  - 활성 트랙 분류: `voice` / `music` / `fullmix` / `silent` (VAD speech ratio + 스펙트럼 대역 + hf slope)
  - 처치 룰 엔진: hum / clipping / DC offset / clicks / phase / channel imbalance 검사 후 RX 플러그인 단축명과 함께 처치 계획 JSON 생성
  - 동일 RMS 신호 그룹 자동 검출 + mirror map (트랙 0 분석 결과를 트랙 1에 미러링)
  - 디렉터리 일괄 모드, `--out-dir`로 영상별 JSON 리포트 저장
  - dry-run only — 실제 처리는 사용자/후속 명령이 결정 (안전한 검토 단계)
  - core 모듈: `audioman.core.obs` (probe_topology, classify_track, diagnose_track, recommend_treatment, dry_run_video)
  - 단위 테스트 17개 (`tests/unit/test_obs.py`)
  - 워크플로우 문서: `docs/obs-workflow.md`

### Changed
- **obs probe**: 기본 동작이 트랙 앞 15초만 보던 것에서 **영상 전체 스캔**으로 변경.
  - OBS 데스크탑 오디오처럼 산발적으로만 신호가 나오는 트랙이 앞 구간 무음으로 silent 오분류되는 문제 해결.
  - `probe_seconds=None`(기본)이면 전체, 명시적으로 숫자를 주면 기존 동작(앞 N초만).
  - CLI: `audioman obs probe --probe-seconds 15` 식으로 명시 가능.
  - 회귀 테스트: `test_probe_topology_full_scan_catches_late_signal`, `test_probe_topology_default_is_full_scan`

## [0.1.0] - 2026-03-26

### Added
- **i18n**: locale 기반 다국어 지원 (`AUDIOMAN_LANG` 환경변수, 시스템 locale 자동 감지)
  - 기본 영어, 한국어 카탈로그 포함, 확장 가능한 구조
- **doctor**: PluginDoctor 스타일 플러그인 분석 엔진
  - 분석 모드: linear, thd, imd, sweep, dynamics, attack-release, waveshaper, performance
  - A/B 비교 (`--compare`), M/S 모드 (`--mid-side`)
  - CLAP 임베딩 프로파일링 (`--clap`, `--clap-sweep`, `--clap-output`)
  - waveshaper v2: 다중 진폭 레벨 + 복수 주기 평균 + 256포인트 리샘플링
  - `--legacy-waveshaper`, `--ws-levels`, `--ws-points` 옵션
- **batch**: `--workers N` 병렬 처리 (process, chain)
- **stream**: 대용량 파일(>500MB) 자동 스트리밍 처리
- **ux**: Rich 프로그레스 바 + ETA (배치 처리)
- **visualize**: Sonic Visualiser SVL export (Vamp 플러그인 + 내장 분석)
- **analyze**: 오디오 분석 (RMS, spectral entropy, silence 감지, ASCII 웨이브폼)
- **fx**: 내장 DSP 이펙트 (normalize, gate, trim, fade, gain)
- Core CLI: scan, list, info, process, chain, preset, dump
- 35개 유닛 테스트 (audio_file, dsp, analysis, preset_manager)

### Fixed
- VST3 플러그인 서브디렉토리 스캔 (`**/*.vst3` glob)
- CLAP 프로파일링 성능 최적화 (플러그인 인스턴스 재사용)
