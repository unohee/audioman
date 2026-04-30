# Changelog

All notable changes to this project will be documented in this file.

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
