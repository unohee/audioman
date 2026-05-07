# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **obs**: OBS Studio 멀티트랙 영상 자동 진단 명령 (`audioman obs probe`, `audioman obs dry-run`)
  - 트랙 토폴로지 분류: `multitrack` / `single` / `duplicated` / `silent`
  - 활성 트랙 분류: `voice` / `music` / `fullmix` / `silent` (VAD speech ratio + 스펙트럼 대역 + hf slope)
  - 처치 룰 엔진: hum / clipping / DC offset / clicks / phase / channel imbalance 검사 후 RX 플러그인 단축명과 함께 처치 계획 JSON 생성
  - 동일 RMS 신호 그룹 자동 검출 + mirror map (트랙 0 분석 결과를 트랙 1에 미러링)
  - 디렉터리 일괄 모드, `--out-dir`로 영상별 JSON 리포트 저장
  - dry-run only — 실제 처리는 사용자/후속 명령이 결정 (안전한 검토 단계)
  - core 모듈: `audioman.core.obs` (probe_topology, classify_track, diagnose_track, recommend_treatment, dry_run_video)
  - 단위 테스트 15개 (`tests/unit/test_obs.py`)
  - 워크플로우 문서: `docs/obs-workflow.md`

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
