#!/usr/bin/env python3
# Created: 2026-03-21
# Purpose: PoC - pedalboard로 iZotope RX 10 VST3 플러그인 호환성 검증
# Dependencies: pedalboard, numpy, soundfile

"""
RX 10 VST3 플러그인 15개에 대해:
1. load_plugin() 로드 가능 여부
2. 파라미터 접근 가능 여부 + 목록 출력
3. 짧은 테스트 오디오 처리 가능 여부
"""

import sys
import json
import traceback
from pathlib import Path

import numpy as np

# RX 10 VST3 플러그인 경로
VST3_DIR = Path("/Library/Audio/Plug-Ins/VST3")
RX10_PLUGINS = [
    "RX 10 Spectral De-noise.vst3",
    "RX 10 Voice De-noise.vst3",
    "RX 10 Guitar De-noise.vst3",
    "RX 10 De-click.vst3",
    "RX 10 De-clip.vst3",
    "RX 10 De-crackle.vst3",
    "RX 10 De-ess.vst3",
    "RX 10 De-hum.vst3",
    "RX 10 De-plosive.vst3",
    "RX 10 De-reverb.vst3",
    "RX 10 Breath Control.vst3",
    "RX 10 Mouth De-click.vst3",
    "RX 10 Repair Assistant.vst3",
    "RX 10 Connect.vst3",
    "RX 10 Monitor.vst3",
]

SAMPLE_RATE = 44100
DURATION = 1.0  # 초


def generate_test_audio() -> np.ndarray:
    """1초 사인파 + 가우시안 노이즈 (float32, mono, shape: (1, samples))"""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), dtype=np.float32)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = 0.1 * np.random.randn(len(t)).astype(np.float32)
    audio = sine + noise
    # pedalboard: (channels, samples)
    return audio.reshape(1, -1)


def test_plugin(plugin_path: Path, test_audio: np.ndarray) -> dict:
    """단일 플러그인 호환성 테스트"""
    result = {
        "name": plugin_path.stem,
        "path": str(plugin_path),
        "loadable": False,
        "parameters": [],
        "param_count": 0,
        "processable": False,
        "error": None,
    }

    # 1) 로드 테스트
    try:
        from pedalboard import load_plugin
        plugin = load_plugin(str(plugin_path))
        result["loadable"] = True
    except Exception as e:
        result["error"] = f"Load failed: {e}"
        return result

    # 2) 파라미터 접근 테스트
    try:
        params = []
        for name, param in plugin.parameters.items():
            info = {
                "name": name,
                "type": type(param).__name__,
            }
            # 값 범위 추출 시도
            try:
                info["value"] = getattr(plugin, name.replace(" ", "_"), None)
                if info["value"] is not None:
                    info["value"] = float(info["value"]) if isinstance(info["value"], (int, float)) else str(info["value"])
            except Exception:
                info["value"] = "N/A"

            try:
                info["range"] = str(param.range)
            except Exception:
                info["range"] = "N/A"

            params.append(info)

        result["parameters"] = params
        result["param_count"] = len(params)
    except Exception as e:
        result["error"] = f"Parameter access failed: {e}"

    # 3) 오디오 처리 테스트
    try:
        output = plugin.process(test_audio, SAMPLE_RATE)
        result["processable"] = True
        result["output_shape"] = list(output.shape)
        result["output_rms"] = float(np.sqrt(np.mean(output**2)))
        result["input_rms"] = float(np.sqrt(np.mean(test_audio**2)))
    except Exception as e:
        result["error"] = f"Process failed: {e}"
        result["processable"] = False

    return result


def main():
    print("=" * 70)
    print("  iZotope RX 10 + pedalboard PoC 호환성 테스트")
    print("=" * 70)
    print(f"pedalboard version: ", end="")
    import pedalboard
    print(pedalboard.__version__)
    print(f"테스트 오디오: {DURATION}s, {SAMPLE_RATE}Hz, mono")
    print()

    test_audio = generate_test_audio()
    results = []

    for plugin_name in RX10_PLUGINS:
        plugin_path = VST3_DIR / plugin_name
        if not plugin_path.exists():
            print(f"  [SKIP] {plugin_name} - 파일 없음")
            results.append({"name": plugin_name, "error": "File not found"})
            continue

        print(f"  테스트 중: {plugin_name}...", end=" ", flush=True)
        result = test_plugin(plugin_path, test_audio)

        status = []
        if result["loadable"]:
            status.append("LOAD:OK")
        else:
            status.append("LOAD:FAIL")
        status.append(f"PARAMS:{result['param_count']}")
        if result["processable"]:
            status.append("PROCESS:OK")
        else:
            status.append("PROCESS:FAIL")

        print(" | ".join(status))
        if result.get("error"):
            print(f"    → {result['error']}")

        results.append(result)

    # 요약 매트릭스
    print()
    print("=" * 70)
    print("  결과 매트릭스")
    print("=" * 70)
    print(f"{'Plugin':<35} {'Load':>6} {'Params':>7} {'Process':>8}")
    print("-" * 70)

    load_ok = process_ok = 0
    for r in results:
        name = r.get("name", "?")[:34]
        load = "O" if r.get("loadable") else "X"
        params = str(r.get("param_count", "?"))
        process = "O" if r.get("processable") else "X"
        print(f"{name:<35} {load:>6} {params:>7} {process:>8}")
        if r.get("loadable"):
            load_ok += 1
        if r.get("processable"):
            process_ok += 1

    print("-" * 70)
    print(f"총 {len(results)}개 | 로드: {load_ok} | 처리: {process_ok}")

    # 상세 파라미터 출력 (로드 성공한 것만)
    print()
    print("=" * 70)
    print("  파라미터 상세 (로드 성공 플러그인)")
    print("=" * 70)
    for r in results:
        if r.get("loadable") and r.get("parameters"):
            print(f"\n[{r['name']}] ({r['param_count']} params)")
            for p in r["parameters"]:
                val = p.get("value", "?")
                rng = p.get("range", "?")
                print(f"  - {p['name']}: value={val}, range={rng}")

    # JSON 출력 저장
    output_path = Path(__file__).parent / "poc_rx_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n상세 결과 저장: {output_path}")


if __name__ == "__main__":
    main()
