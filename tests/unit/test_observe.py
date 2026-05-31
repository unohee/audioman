# Created: 2026-05-11
# Purpose: `audioman observe` 명령 통합 — JSON envelope, 메타 필드, finding[] 보장.

import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf


SR = 48000


def _make_faulty_wav(path):
    t = np.linspace(0, 1.0, SR, endpoint=False, dtype=np.float32)
    sig = np.clip(2.0 * np.sin(2 * np.pi * 1000 * t), -1.0, 1.0)
    sf.write(str(path), sig, SR)


def _run_observe(path, *extra, env_extra=None):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(
        [sys.executable, "-m", "audioman", "--json", "--plain", "observe", str(path), *extra],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


class TestObserveEnvelope:
    def test_required_meta_fields(self, tmp_path):
        p = tmp_path / "clip.wav"
        _make_faulty_wav(p)
        payload = _run_observe(p)

        # 후기 #3 직접 검증: duration/total_samples 항상 채워진다.
        assert payload["duration_sec"] is not None
        assert payload["total_samples"] == SR
        assert payload["sample_rate"] == SR
        assert payload["channels"] == 1
        assert payload["$schema"] == "audioman://schema/finding.v1.json"
        assert "audioman_version" in payload
        assert payload["command"] == "observe"

    def test_clip_finding_present(self, tmp_path):
        p = tmp_path / "clip.wav"
        _make_faulty_wav(p)
        payload = _run_observe(p)
        codes = {f["code"] for f in payload["findings"]}
        assert "CLIP_SAMPLE_PEAK_EXCEEDED" in codes

    def test_summary_counts_match_findings(self, tmp_path):
        p = tmp_path / "clip.wav"
        _make_faulty_wav(p)
        payload = _run_observe(p)
        sev_counts = payload["summary"]["by_severity"]
        assert sum(sev_counts.values()) == payload["summary"]["total"]
        assert payload["summary"]["total"] == len(payload["findings"])

    def test_category_filter(self, tmp_path):
        p = tmp_path / "clip.wav"
        _make_faulty_wav(p)
        payload = _run_observe(p, "--category", "spectral")
        # signal 카테고리가 disable됐으므로 clipping finding이 없어야 한다.
        assert all(f["category"] == "spectral" for f in payload["findings"])

    def test_severity_filter(self, tmp_path):
        p = tmp_path / "clip.wav"
        _make_faulty_wav(p)
        payload = _run_observe(p, "--severity", "critical")
        assert all(f["severity"] == "critical" for f in payload["findings"])
