# tests/unit/test_voiceover.py — voiceover batch processing (VAD → denoise → LUFS leveling).
#
# No VST3 plugins exist on this host (AUD-1857), so the denoise plugin is faked.
# VAD is monkeypatched: silero's actual decisions are irrelevant to the code under
# test (segment bookkeeping, denoise dispatch, leveling delegation, reporting).

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from audioman.core import voiceover
from audioman.core.vad import Segment

SR = 48000


def _tone(duration=2.0, sr=SR, amp=0.25):
    t = np.arange(int(duration * sr)) / sr
    return (amp * np.sin(2 * np.pi * 300 * t)).astype(np.float32)


@pytest.fixture
def speech_wav(tmp_path):
    mono = _tone(2.0)
    stereo = np.stack([mono, mono])
    path = tmp_path / "vo.wav"
    sf.write(str(path), stereo.T, SR, subtype="FLOAT")
    return path


@pytest.fixture
def segments():
    return [Segment(start=SR // 2, end=SR, kind="speech")]


class TestVoiceoverResultDict:
    def test_to_dict_rounds_and_renames(self):
        r = voiceover.VoiceoverResult(
            input_path="/in.wav", output_path="/out.wav", sample_rate=48000,
            duration_sec=1.23456, n_speech_segments=2, speech_total_sec=0.98765,
            noise_total_sec=0.34567, denoise_plugin="RX 10 Voice De-noise",
            leveling_meta={"k": 1}, measured_in={"integrated_lufs": -20.0},
            measured_out={"integrated_lufs": -20.0},
        )
        d = r.to_dict()
        assert d["input"] == "/in.wav"
        assert d["output"] == "/out.wav"
        assert d["duration_sec"] == 1.235
        assert d["speech_total_sec"] == 0.988
        assert d["denoise_plugin"] == "RX 10 Voice De-noise"
        assert d["leveling"] == {"k": 1}


class TestAnalyze:
    def test_reports_segment_stats(self, speech_wav, monkeypatch):
        monkeypatch.setattr(
            voiceover, "detect_speech",
            lambda audio, sr, **kw: [Segment(0, sr, "speech")],
        )
        report = voiceover.analyze(speech_wav)
        assert report["n_speech_segments"] == 1
        assert report["speech_total_sec"] == pytest.approx(1.0, abs=0.01)
        assert report["noise_total_sec"] == pytest.approx(1.0, abs=0.01)
        assert report["speech_ratio"] == pytest.approx(0.5, abs=0.01)
        assert report["sample_rate"] == SR
        assert "loudness" in report

    def test_full_speech_has_zero_noise(self, speech_wav, monkeypatch):
        monkeypatch.setattr(
            voiceover, "detect_speech",
            lambda audio, sr, **kw: [Segment(0, audio.shape[-1], "speech")],
        )
        report = voiceover.analyze(speech_wav)
        assert report["noise_total_sec"] == 0.0
        assert report["speech_ratio"] == 1.0


class TestApplyDenoise:
    def test_unknown_plugin_raises(self, monkeypatch):
        monkeypatch.setattr(voiceover, "get_registry",
                            lambda: type("R", (), {"get": lambda s, n: None})())
        with pytest.raises(ValueError, match="Plugin not found"):
            voiceover._apply_denoise(np.ones((2, 100), dtype=np.float32), SR, "missing", None)

    def test_applies_wrapper_and_returns_full_name(self, monkeypatch):
        from audioman.plugins.parameter import PluginMeta
        meta = PluginMeta(name="RX 10 Voice De-noise", short_name="voice-de-noise",
                          path="/rx.vst3", format="vst3")
        made = []

        class _W:
            def __init__(self, path):
                self.path = path
                self.params = None
                made.append(self)

            def load(self):
                pass

            def set_parameters(self, p):
                self.params = p

            def process(self, audio, sr):
                return audio * 0.5

        monkeypatch.setattr(voiceover, "get_registry",
                            lambda: type("R", (), {"get": lambda s, n: meta})())
        monkeypatch.setattr(voiceover, "VST3PluginWrapper", _W)

        out, name = voiceover._apply_denoise(
            np.ones((2, 10), dtype=np.float32), SR, "voice-de-noise", {"reduction": 12}
        )
        np.testing.assert_allclose(out, 0.5)
        assert name == "RX 10 Voice De-noise"
        assert made[0].params == {"reduction": 12}

    def test_no_params_leaves_wrapper_default(self, monkeypatch):
        from audioman.plugins.parameter import PluginMeta
        meta = PluginMeta(name="N", short_name="n", path="/n.vst3", format="vst3")
        made = []

        class _W:
            def __init__(self, path):
                self.params = None
                made.append(self)

            def load(self):
                pass

            def set_parameters(self, p):
                self.params = p

            def process(self, audio, sr):
                return audio

        monkeypatch.setattr(voiceover, "get_registry",
                            lambda: type("R", (), {"get": lambda s, n: meta})())
        monkeypatch.setattr(voiceover, "VST3PluginWrapper", _W)
        voiceover._apply_denoise(np.ones((2, 10), dtype=np.float32), SR, "n", None)
        assert made[0].params is None


class TestProcess:
    def test_process_with_denoise(self, speech_wav, tmp_path, monkeypatch):
        from audioman.plugins.parameter import PluginMeta
        meta = PluginMeta(name="RX 10 Voice De-noise", short_name="voice-de-noise",
                          path="/rx.vst3", format="vst3")

        class _W:
            def __init__(self, path):
                pass

            def load(self):
                pass

            def set_parameters(self, p):
                pass

            def process(self, audio, sr):
                return audio * 0.9

        monkeypatch.setattr(voiceover, "get_registry",
                            lambda: type("R", (), {"get": lambda s, n: meta})())
        monkeypatch.setattr(voiceover, "VST3PluginWrapper", _W)
        monkeypatch.setattr(voiceover, "detect_speech",
                            lambda audio, sr, **kw: [Segment(0, audio.shape[-1], "speech")])

        out_path = tmp_path / "out" / "vo_out.wav"
        result = voiceover.process(speech_wav, out_path, denoise_plugin="voice-de-noise")

        assert out_path.exists()  # parent dir created
        assert result.denoise_plugin == "RX 10 Voice De-noise"
        assert result.n_speech_segments == 1
        assert result.leveling_meta is not None
        assert result.measured_out is not None
        assert result.to_dict()["output"] == str(out_path)

    def test_process_without_denoise(self, speech_wav, tmp_path, monkeypatch):
        monkeypatch.setattr(voiceover, "detect_speech",
                            lambda audio, sr, **kw: [Segment(0, audio.shape[-1], "speech")])
        result = voiceover.process(
            speech_wav, tmp_path / "vo.wav", denoise_plugin=None
        )
        assert result.denoise_plugin is None
        assert result.measured_out is not None

    def test_process_no_speech_segments(self, speech_wav, tmp_path, monkeypatch):
        monkeypatch.setattr(voiceover, "detect_speech", lambda audio, sr, **kw: [])
        result = voiceover.process(
            speech_wav, tmp_path / "vo.wav", denoise_plugin=None
        )
        assert result.n_speech_segments == 0
        assert result.speech_total_sec == 0.0
        assert result.noise_total_sec > 0.0
