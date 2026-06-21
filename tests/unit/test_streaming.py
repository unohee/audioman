# Created: 2026-05-31
# Purpose: 블록 스트리밍 엔진 + 클릭 triage + RT 벤치 테스트.
# pedalboard 빌트인만 사용 — VST3 플러그인 불필요.

import numpy as np
import pytest

from audioman.core.streaming import (
    render_offline,
    render_streamed,
    make_pedalboard_process_fn,
)
from audioman.core.discontinuity import (
    detect_discontinuities,
    detect_nonfinite,
    null_test,
)
from audioman.core.rt_bench import benchmark


SR = 48000


def _sine(freq=440.0, dur=0.5, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32).reshape(1, -1)


def _reverb_fn():
    from pedalboard import Pedalboard, Reverb
    return make_pedalboard_process_fn(Pedalboard([Reverb(room_size=0.8, wet_level=0.5)]))


# --- streaming engine -------------------------------------------------------

def test_streamed_matches_offline_bit_for_bit():
    """올바른 스트리밍(reset=False 연속)은 오프라인 렌더와 비트 단위로 일치해야 한다."""
    x = _sine()
    off = render_offline(x, SR, _reverb_fn())
    st = render_streamed(x, SR, _reverb_fn(), block_size=512, reset_per_block=False)
    n = min(off.audio.shape[1], st.audio.shape[1])
    max_db = 20 * np.log10(np.max(np.abs(off.audio[:, :n] - st.audio[:, :n])) + 1e-30)
    assert max_db < -120.0, f"streamed should match offline, got {max_db:.1f} dB"


def test_reset_per_block_breaks_continuity():
    """매 블록 reset하면 오프라인과 유의미하게 갈린다 (클릭 재현)."""
    x = _sine()
    off = render_offline(x, SR, _reverb_fn())
    bad = render_streamed(x, SR, _reverb_fn(), block_size=512, reset_per_block=True)
    n = min(off.audio.shape[1], bad.audio.shape[1])
    max_db = 20 * np.log10(np.max(np.abs(off.audio[:, :n] - bad.audio[:, :n])) + 1e-30)
    assert max_db > -30.0, f"reset-per-block should diverge, got {max_db:.1f} dB"


def test_block_count_and_length():
    """블록 수와 출력 길이가 입력과 일치해야 한다."""
    x = _sine(dur=1.0)
    st = render_streamed(x, SR, _reverb_fn(), block_size=512)
    expected_blocks = -(-x.shape[1] // 512)  # ceil
    assert len(st.timings) == expected_blocks
    assert st.audio.shape[1] == x.shape[1]


def test_invalid_block_size():
    with pytest.raises(ValueError):
        render_streamed(_sine(), SR, _reverb_fn(), block_size=0)


# --- discontinuity / triage -------------------------------------------------

def test_clean_sine_no_false_positives():
    """깨끗한 사인파에서 클릭을 검출하면 안 된다."""
    x = _sine()
    findings = detect_discontinuities(x, SR, block_size=512)
    assert findings == []


def test_block_aligned_click_detected():
    """블록 경계에 주입한 step은 block_aligned=True, critical로 분류된다."""
    x = _sine().copy()
    bs = 512
    x[0, bs * 10] += 0.4  # 정확히 블록 경계
    findings = detect_discontinuities(x, SR, block_size=bs)
    aligned = [f for f in findings if f.measurement["block_aligned"]]
    assert len(aligned) >= 1
    f = aligned[0]
    assert f.severity.value == "critical"
    assert f.measurement["nearest_block_edge"] == bs * 10


def test_unaligned_click_is_warn_not_critical():
    """블록 경계가 아닌 곳의 클릭은 source 결함(warn)으로 분류."""
    x = _sine().copy()
    bs = 512
    pos = bs * 10 + 137  # 경계에서 충분히 떨어진 위치
    x[0, pos] += 0.4
    findings = detect_discontinuities(x, SR, block_size=bs)
    matched = [f for f in findings if abs(f.where.start_sample - pos) <= 2]
    assert matched, "click should be detected"
    assert matched[0].measurement["block_aligned"] is False
    assert matched[0].severity.value == "warn"


def test_nonfinite_detection():
    x = _sine().copy()
    x[0, 1000] = np.nan
    x[0, 2000] = np.inf
    findings = detect_nonfinite(x, SR)
    assert len(findings) == 1
    assert findings[0].measurement["nonfinite_samples"] == 2
    assert findings[0].code.value == "NONFINITE_SAMPLES"


def test_null_test_identical_is_empty():
    x = _sine()
    assert null_test(x, x, SR) == []


def test_null_test_detects_divergence():
    x = _sine()
    y = x.copy()
    y[0, 5000] += 0.5
    findings = null_test(x, y, SR)
    assert len(findings) == 1
    assert findings[0].measurement["max_diff_db"] > -60.0


def test_null_test_latency_compensation():
    """latency_samples만큼 어긋난 동일 신호는 보상 후 일치해야 한다."""
    x = _sine()
    shifted = np.concatenate([np.zeros((1, 64), dtype=np.float32), x], axis=1)
    # 보상 없이는 차이가 큼, 보상하면 일치
    assert null_test(x, shifted, SR) != []
    assert null_test(x, shifted, SR, latency_samples=64) == []


# --- RT bench ---------------------------------------------------------------

def test_benchmark_basic_fields():
    x = _sine(dur=1.0)
    st = render_streamed(x, SR, _reverb_fn(), block_size=512)
    rep = benchmark(st)
    assert rep.block_size == 512
    assert rep.blocks > 0
    assert rep.deadline_ms == pytest.approx(512 / SR * 1000, abs=1e-3)
    assert rep.rt_factor_max >= rep.rt_factor_p99 >= rep.rt_factor_p50
    assert rep.est_max_tracks >= 1


def test_benchmark_rt_factor_monotonic_with_load():
    """무거운 처리가 가벼운 처리보다 RT factor가 커야 한다."""
    x = _sine(dur=1.0)

    def light(b, sr, reset):
        return b * 0.5

    def heavy(b, sr, reset):
        for _ in range(30):
            np.fft.rfft(b, axis=1)
        return b * 0.5

    rl = benchmark(render_streamed(x, SR, light, block_size=512))
    rh = benchmark(render_streamed(x, SR, heavy, block_size=512))
    assert rh.rt_factor_mean > rl.rt_factor_mean


def test_benchmark_smaller_block_higher_rt_factor():
    """작은 블록일수록 RT factor가 크다 (고정 오버헤드 / 짧은 deadline)."""
    x = _sine(dur=1.0)

    def heavy(b, sr, reset):
        for _ in range(30):
            np.fft.rfft(b, axis=1)
        return b * 0.5

    small = benchmark(render_streamed(x, SR, heavy, block_size=64))
    large = benchmark(render_streamed(x, SR, heavy, block_size=1024))
    assert small.rt_factor_mean > large.rt_factor_mean
