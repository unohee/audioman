# tests/unit/test_waveform.py — ASCII 웨이브폼 렌더러 테스트

import numpy as np
import pytest

from audioman.core.waveform import (
    render_waveform,
    render_envelope,
    render_spectral_envelope,
    _make_time_axis,
)


class TestRenderWaveform:
    def test_returns_string(self):
        audio = np.random.randn(2, 44100).astype(np.float32)
        result = render_waveform(audio, sample_rate=44100)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_header_info(self):
        audio = 0.5 * np.ones((2, 44100), dtype=np.float32)
        result = render_waveform(audio, sample_rate=44100)
        assert "peak:" in result
        assert "dB" in result
        assert "44100Hz" in result

    def test_width_controls_columns(self):
        audio = np.random.randn(1, 44100).astype(np.float32)
        result = render_waveform(audio, sample_rate=44100, width=40, height=8)
        lines = result.strip().split("\n")
        # 중앙선은 width 길이
        center_line = [l for l in lines if "─" in l]
        assert len(center_line) > 0
        assert len(center_line[0]) == 40

    def test_mono_input(self):
        mono = np.random.randn(22050).astype(np.float32)
        result = render_waveform(mono, sample_rate=44100, width=20, height=4)
        assert isinstance(result, str)

    def test_silent_audio(self):
        audio = np.zeros((2, 44100), dtype=np.float32)
        result = render_waveform(audio, sample_rate=44100, width=20, height=4)
        assert "peak: 0.000" in result

    def test_peak_mode(self):
        audio = np.random.randn(2, 44100).astype(np.float32)
        result = render_waveform(audio, sample_rate=44100, mode="peak", width=20, height=4)
        assert "mode: peak" in result

    def test_rms_mode_default(self):
        audio = np.random.randn(2, 44100).astype(np.float32)
        result = render_waveform(audio, sample_rate=44100, width=20, height=4)
        assert "mode: rms" in result

    def test_line_count(self):
        """header + height + 1(중앙선) + 1(시간축) 줄"""
        audio = np.random.randn(2, 44100).astype(np.float32)
        height = 8
        result = render_waveform(audio, sample_rate=44100, width=20, height=height)
        lines = result.strip().split("\n")
        # header(1) + 상단(height/2) + 중앙(1) + 하단(height/2) + 시간축(1)
        expected = 1 + height // 2 + 1 + height // 2 + 1
        assert len(lines) == expected

    @pytest.mark.parametrize("kwargs", [
        {"height": 1},
        {"width": 7},
        {"mode": "unknown"},
    ])
    def test_rejects_unrenderable_parameters(self, kwargs):
        with pytest.raises(ValueError):
            render_waveform(np.ones(32, dtype=np.float32), sample_rate=44100, **kwargs)

    def test_trailing_peak_is_rendered(self):
        audio = np.zeros(10, dtype=np.float32)
        audio[-1] = 1.0
        result = render_waveform(audio, sample_rate=44100, width=8, height=4, mode="peak")
        assert "peak: 1.000" in result


class TestRenderEnvelope:
    def test_returns_string(self):
        audio = np.random.randn(2, 44100).astype(np.float32)
        result = render_envelope(audio, sample_rate=44100)
        assert isinstance(result, str)

    def test_contains_rms_info(self):
        audio = np.random.randn(2, 44100).astype(np.float32)
        result = render_envelope(audio, sample_rate=44100)
        assert "rms peak:" in result
        assert "dB" in result

    def test_mono_input(self):
        mono = np.random.randn(44100).astype(np.float32)
        result = render_envelope(mono, sample_rate=44100, width=20, height=4)
        assert isinstance(result, str)

    def test_silent_audio(self):
        audio = np.zeros((2, 44100), dtype=np.float32)
        result = render_envelope(audio, sample_rate=44100, width=20, height=4)
        assert "rms peak:" in result

    def test_line_count(self):
        audio = np.random.randn(2, 44100).astype(np.float32)
        height = 6
        result = render_envelope(audio, sample_rate=44100, width=20, height=height)
        lines = result.strip().split("\n")
        # header(1) + height + 시간축(1)
        assert len(lines) == 1 + height + 1


class TestRenderSpectralEnvelope:
    def test_returns_string(self):
        centroid = [500.0, 600.0, 700.0, 800.0]
        entropy = [3.0, 3.5, 4.0, 3.2]
        result = render_spectral_envelope(
            centroid, entropy,
            sample_rate=44100, duration=1.0, width=20, height=4,
        )
        assert isinstance(result, str)
        assert "centroid" in result
        assert "entropy" in result

    def test_units_shown(self):
        result = render_spectral_envelope(
            [100.0, 200.0], [1.0, 2.0],
            sample_rate=44100, duration=1.0, width=20, height=4,
        )
        assert "Hz" in result
        assert "bits" in result

    def test_empty_values(self):
        result = render_spectral_envelope(
            [], [],
            sample_rate=44100, duration=1.0, width=20, height=4,
        )
        # 빈 입력 → 빈 결과
        assert "centroid" not in result

    def test_single_metric(self):
        result = render_spectral_envelope(
            [500.0, 600.0], [],
            sample_rate=44100, duration=1.0, width=20, height=4,
        )
        assert "centroid" in result
        assert "entropy" not in result


class TestMakeTimeAxis:
    def test_basic(self):
        result = _make_time_axis(40, 2.5)
        assert result.startswith("0s")
        assert "2.5s" in result

    def test_width(self):
        result = _make_time_axis(60, 1.0)
        # 라벨 포함해서 적절한 길이
        assert len(result) > 0


class TestEmptyChunkPadding:
    def test_more_columns_than_samples_pads_with_silence(self):
        # width 200 > sample count 16 → trailing columns hit the empty-chunk branch
        audio = np.ones(16, dtype=np.float32) * 0.5
        out = render_waveform(audio, sample_rate=48000, width=200, height=8)
        assert isinstance(out, str)
        assert len(out) > 0

    def test_empty_chunk_in_envelope_renderer(self):
        audio = np.ones(16, dtype=np.float32) * 0.5
        out = render_envelope(audio, sample_rate=48000, width=200)
        assert isinstance(out, str)


class TestValidateRenderParams:
    def test_zero_width_rejected(self):
        from audioman.core.waveform import _validate_render_params
        with pytest.raises(ValueError, match="width must be positive"):
            _validate_render_params(0, 16, 44100)

    def test_zero_sample_rate_rejected(self):
        from audioman.core.waveform import _validate_render_params
        # width >= 8 and height >= 2 so validation reaches the sample-rate check
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            _validate_render_params(80, 16, 0)
