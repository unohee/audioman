# Created: 2026-03-25
# Purpose: i18n 지원 — locale 감지, 기본 영어 + 한국어, 확장 가능한 구조
# Dependencies: locale, os (stdlib only)

from __future__ import annotations

import locale
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# 메시지 카탈로그
# 기본값(영어)은 키 자체이므로 영어 카탈로그는 불필요.
# 새 언어 추가: CATALOGS["lang_code"] = { "english string": "translated" }
# ---------------------------------------------------------------------------

CATALOGS: dict[str, dict[str, str]] = {}

# --- 한국어 ---
CATALOGS["ko"] = {
    # app.py
    "Cross-platform CLI wrapper for VST3/AU audio plugins": "VST3/AU 오디오 플러그인을 위한 크로스플랫폼 CLI 래퍼",
    "JSON output mode": "JSON 출력 모드",
    "Verbose logging": "상세 로깅",
    "Available commands": "사용 가능한 명령",
    # scan
    "Scan system for VST3/AU plugins": "시스템에서 VST3/AU 플러그인 검색",
    "Additional search paths": "추가 검색 경로",
    "Ignore cache and rescan": "캐시 무시하고 재스캔",
    # list
    "List registered plugins": "등록된 플러그인 목록",
    "Format filter": "포맷 필터",
    "Vendor filter": "벤더 필터",
    # info
    "Plugin details + parameter list": "플러그인 상세 정보 + 파라미터 목록",
    "Plugin name (short_name or alias)": "플러그인 이름 (short_name 또는 별칭)",
    # process
    "Process audio with a single plugin": "단일 플러그인으로 오디오 처리",
    "Input audio file or directory": "입력 오디오 파일 또는 디렉토리",
    "Plugin name": "플러그인 이름",
    "Parameter (key=value)": "파라미터 (key=value)",
    "Output file or directory": "출력 파일 또는 디렉토리",
    "Number of passes (2=adaptive learning multi-pass)": "처리 횟수 (2=adaptive 학습용 멀티패스)",
    "Include subdirectories (batch)": "하위 디렉토리 포함 (배치)",
    "Output filename suffix (batch)": "출력 파일명 접미사 (배치)",
    "Show plan without executing": "실행하지 않고 계획만 표시",
    "Number of parallel workers (default: 1)": "병렬 처리 워커 수 (기본: 1)",
    # chain
    "Process audio through multiple plugins sequentially": "다중 플러그인 순차 처리",
    "Processing chain (e.g. 'dehum:notch_frequency=60,declick,denoise:noise_reduction_db=15')": "처리 체인 (예: 'dehum:notch_frequency=60,declick,denoise:noise_reduction_db=15')",
    # preset
    "Preset management": "프리셋 관리",
    "Save preset": "프리셋 저장",
    "Preset name": "프리셋 이름",
    "Description": "설명",
    "Show preset info": "프리셋 정보 표시",
    "Plugin name (optional)": "플러그인 이름 (선택)",
    "List presets": "프리셋 목록",
    "Plugin filter": "플러그인 필터",
    "Delete preset": "프리셋 삭제",
    # dump
    "Dump plugin parameter state to JSON/JSONL": "플러그인 파라미터 상태를 JSON/JSONL로 덤프",
    "Plugin name (omit for --all)": "플러그인 이름 (생략 시 --all 필요)",
    "Set parameter before dump (key=value)": "덤프 전 파라미터 설정 (key=value)",
    "Preset name (apply before dump)": "프리셋 이름 (적용 후 덤프)",
    "Save dump as preset": "덤프 결과를 프리셋으로 저장",
    "Dump all plugins as JSONL": "모든 플러그인 JSONL 덤프",
    "Plugin name filter (with --all)": "플러그인 이름 필터 (--all과 함께)",
    "Format filter (with --all)": "포맷 필터 (--all과 함께)",
    "JSONL output file (default: stdout)": "JSONL 출력 파일 (기본: stdout)",
    # analyze
    "Audio analysis (RMS, spectral entropy, silence detection, etc.)": "오디오 분석 (RMS, spectral entropy, silence 감지 등)",
    "Per-frame detailed output": "프레임 단위 상세 출력",
    "Frame size (default: 2048)": "프레임 크기 (기본: 2048)",
    "Hop size (default: 512)": "홉 크기 (기본: 512)",
    "Silence detection threshold dB (default: -40)": "Silence 감지 임계값 dB (기본: -40)",
    "Show ASCII waveform": "ASCII 웨이브폼 표시",
    "Waveform width (default: 80)": "웨이브폼 가로 폭 (기본: 80)",
    "Waveform height (default: 16)": "웨이브폼 세로 높이 (기본: 16)",
    "Waveform mode (default: peak)": "웨이브폼 모드 (기본: peak)",
    # fx
    "Built-in DSP effects (fade, trim, normalize, gate, gain)": "내장 DSP 이펙트 (fade, trim, normalize, gate, gain)",
    "Effect type": "이펙트 종류",
    "Linear fade in": "선형 fade in",
    "Fade length (samples)": "fade 길이 (샘플)",
    "Fade length (seconds)": "fade 길이 (초)",
    "Output path": "출력 경로",
    "Linear fade out": "선형 fade out",
    "Trim by samples/time": "샘플/시간 단위 트리밍",
    "Start sample": "시작 샘플",
    "End sample": "끝 샘플",
    "Start (seconds)": "시작 (초)",
    "End (seconds)": "끝 (초)",
    "Trim leading/trailing silence": "앞뒤 silence 제거",
    "Threshold dB (default: -40)": "임계값 dB (기본: -40)",
    "Silence boundary padding samples": "silence 경계 패딩 샘플",
    "Normalize (peak or RMS)": "정규화 (peak 또는 RMS)",
    "Peak target dB (e.g. -1)": "피크 목표 dB (예: -1)",
    "RMS target dB (e.g. -20)": "RMS 목표 dB (예: -20)",
    "Noise gate (RMS-based)": "노이즈 게이트 (RMS 기반)",
    "Threshold dB (default: -50)": "임계값 dB (기본: -50)",
    "Attack time (seconds)": "attack 시간 (초)",
    "Release time (seconds)": "release 시간 (초)",
    "dB gain": "dB 게인",
    "Gain (dB)": "게인 (dB)",
    # visualize
    "Vamp plugin or built-in analysis -> Sonic Visualiser SVL file": "Vamp 플러그인 또는 내장 분석 → Sonic Visualiser SVL 파일 생성",
    "Built-in analysis type": "내장 분석 타입",
    "Input audio file": "입력 오디오 파일",
    "Vamp plugin ID (e.g. qm-vamp-plugins:qm-chromagram)": "Vamp 플러그인 ID (예: qm-vamp-plugins:qm-chromagram)",
    "Output SVL file path (default: auto)": "출력 SVL 파일 경로 (기본: 자동 생성)",
    "Vamp plugin output name (for multiple outputs)": "Vamp 플러그인 출력 이름 (복수 출력 시)",
    "FFT frame size (default: 2048)": "FFT 프레임 크기 (기본: 2048)",
    "List installed Vamp plugins": "설치된 Vamp 플러그인 목록",
    "Query plugin output info": "플러그인 출력 정보 조회",
    "Open in Sonic Visualiser after creation": "생성 후 Sonic Visualiser로 열기",
    # doctor
    "Plugin analysis — frequency response, THD, dynamics, waveshaper, performance": "플러그인 분석 — frequency response, THD, dynamics, waveshaper, performance",
    "Plugin name or path": "플러그인 이름 또는 경로",
    "Analysis mode (default: all)": "분석 모드 (기본: all)",
    "Test frequency Hz": "테스트 주파수 Hz",
    "Input level dB": "입력 레벨 dB",
    "M/S mode": "M/S 모드",
    "Compare with second plugin": "2번째 플러그인과 비교",
    "Second plugin parameters": "2번째 플러그인 파라미터",
    "CLAP embedding profiling (per-parameter saturation fingerprint)": "CLAP 임베딩 프로파일링 (파라미터별 새추레이션 지문)",
    "CLAP sweep parameters (e.g. --clap-sweep drive=0,25,50,75,100)": "CLAP 스윕 파라미터 (예: --clap-sweep drive=0,25,50,75,100)",
    "CLAP embedding npy save path": "CLAP 임베딩 npy 저장 경로",
    "Save result JSON file": "결과 JSON 파일 저장",
}


# ---------------------------------------------------------------------------
# Locale 감지 및 번역 함수
# ---------------------------------------------------------------------------

def _detect_lang() -> str:
    """AUDIOMAN_LANG > LC_ALL > LC_MESSAGES > LANG 순서로 언어 코드 감지."""
    # 환경변수로 직접 지정 가능
    env_lang = os.environ.get("AUDIOMAN_LANG", "")
    if env_lang:
        return env_lang.split("_")[0].split("-")[0].lower()

    # 시스템 locale
    try:
        loc = locale.getlocale()[0] or locale.getdefaultlocale()[0] or ""
    except ValueError:
        loc = ""

    if loc:
        return loc.split("_")[0].lower()

    return "en"


_current_lang: str | None = None


def get_lang() -> str:
    """현재 활성 언어 코드 반환."""
    global _current_lang
    if _current_lang is None:
        _current_lang = _detect_lang()
    return _current_lang


def set_lang(lang: str) -> None:
    """언어를 수동으로 설정."""
    global _current_lang
    _current_lang = lang.split("_")[0].split("-")[0].lower()


def _(msg: str) -> str:
    """메시지를 현재 locale에 맞게 번역. 번역이 없으면 원문(영어) 반환."""
    lang = get_lang()
    if lang == "en":
        return msg
    catalog = CATALOGS.get(lang)
    if catalog is None:
        return msg
    return catalog.get(msg, msg)
