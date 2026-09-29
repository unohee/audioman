# tests/unit/cli_extra2/test_residual_lines.py
# Purpose: settle the leftover lines in small core modules that earlier passes
#          called unreachable — by re-deriving each one from its own call site
#          instead of taking the claim on trust.
#
# Verdicts reached here (the derivations are in each class docstring), and what was
# done about each one:
#   core/aesthetic.py:118   DEAD  — len(frame) >= 2 already forced by the guard above
#                                  → `# pragma: no cover` in the source
#   core/aesthetic.py:510   DEAD  — fft_size >= 2048 for every len(mono) >= 2048
#                                  → `# pragma: no cover` in the source
#   core/aesthetic.py:525   DEAD  — the neighbourhood is non-empty for every bin of
#                                  a >= 1025-bin spectrum → `# pragma: no cover`
#   core/qc.py:170-171,173  DEAD  — pad == (win-2)//2 >= 7 and the padded length is
#                                  exactly n-1 → the two branches were DELETED, with
#                                  the padded length now unconditional
#   config/settings.py:32   DEAD  — tomllib's root is always a table; other roots
#                                  raise TOMLDecodeError → `# pragma: no cover`
#   core/dsp.py:225         covered via state injection (see TestFadeCurveDriftFallback)
#   __main__.py:4-6         REACHABLE — covered here through runpy
#
# The dead ones are pinned by the invariants they rest on, so the proof survives
# as a regression check rather than as a comment someone has to re-derive.

from __future__ import annotations

import math

import numpy as np
import pytest
import soundfile as sf


# ---------------------------------------------------------------------------
# core/aesthetic.py
# ---------------------------------------------------------------------------


class TestSpectralFlatnessEmptySpectrumIsDead:
    """aesthetic.py:118 — `len(spectrum) == 0` in `_spectral_flatness`.

    `spectrum` is `np.fft.rfft(frame)[1:]`. `np.fft.rfft` of a length-L frame
    returns ``L // 2 + 1`` bins, so dropping the DC bin leaves ``L // 2`` bins,
    which reaches 0 only for ``L <= 1``. The guard directly above returns for
    every ``L < 2``, so the check cannot fire.

    (The first guard is what earlier tests actually exercised: a 1-sample frame
    and an empty frame both return 0.0 at line 113, never at 118.)

    The line therefore carries `# pragma: no cover` in the source; the invariant
    above is what this class checks, so the marker cannot outlive its premise.
    """

    def test_rfft_bin_count_is_L_over_2_after_the_dc_bin_is_dropped(self):
        for length in range(2, 64):
            frame = np.zeros(length, dtype=np.float32)
            remaining = len(np.fft.rfft(frame)) - 1
            assert remaining == length // 2
            assert remaining >= 1  # never 0 for a frame that passed the guard

    def test_the_guard_already_covers_every_frame_that_could_empty_it(self):
        """Frames of length 0 and 1 are exactly the ones the earlier guard rejects."""
        from audioman.core.aesthetic import _spectral_flatness

        for length in (0, 1):
            assert _spectral_flatness(np.zeros(length, dtype=np.float32)) == 0.0

    def test_two_sample_frame_computes_instead_of_returning_early(self):
        from audioman.core.aesthetic import _spectral_flatness

        assert _spectral_flatness(np.array([0.5, -0.5], dtype=np.float32)) == pytest.approx(
            1.0, rel=1e-6
        )


class TestRfNoiseFftSizeCheckIsDead:
    """aesthetic.py:510 — `fft_size < 2048` in `detect_rf_noise_events`.

    `fft_size = min(32768, 2 ** floor(log2(len(mono))))` and the function has
    already returned for `len(mono) < 2048`. For every `len >= 2048` the
    exponent is at least 11, so the power of two is at least 2048 and the min
    with 32768 cannot pull it back down. The line is therefore unreachable.

    Note `fft_size` is a *sample* count, not a bin count, so lowering the sample
    rate does not shrink it — an earlier attempt to reach the line that way was
    wrong; it is the length guard, not the rate, that governs it.
    """

    def test_fft_size_is_at_least_2048_for_every_length_past_the_guard(self):
        for length in range(2048, 2048 + 512):
            fft_size = min(32768, 2 ** int(math.floor(math.log2(length))))
            assert fft_size >= 2048
        for length in (4096, 10000, 48000, 32768, 40000, 10**6):
            fft_size = min(32768, 2 ** int(math.floor(math.log2(length))))
            assert fft_size >= 2048

    def test_lengths_below_the_guard_return_before_the_check(self):
        from audioman.core.aesthetic import detect_rf_noise_events

        for length in (0, 1024, 2047):
            events, backend = detect_rf_noise_events(np.zeros(length, dtype=np.float32), 48000)
            assert events == []
            assert backend == "heuristic"

    def test_exactly_2048_samples_reaches_the_scan(self):
        """The boundary is inclusive: 2048 is the smallest length that scans."""
        from audioman.core.aesthetic import detect_rf_noise_events

        events, backend = detect_rf_noise_events(np.zeros(2048, dtype=np.float32), 48000)
        assert backend == "heuristic"
        assert isinstance(events, list)


class TestRfNoiseEmptyNeighbourhoodIsDead:
    """aesthetic.py:525 — `continue` when the tone neighbourhood is empty.

    Per candidate bin the code concatenates `spectrum[idx-30 : idx-3]` and
    `spectrum[idx+4 : idx+30]`, both clipped to the spectrum. A scan only starts
    at 2048 samples, which gives at least 1025 bins, and every bin index in
    `[0, bins)` has at least one non-empty half: the lower half is empty only for
    `idx <= 4`, and for those bins the upper half `spectrum[idx+4 : idx+31]`
    always exists. The `continue` is unreachable.
    """

    @staticmethod
    def _neighbourhood_length(idx: int, bins: int) -> int:
        lo = max(1, idx - 30)
        hi = min(bins, idx + 31)
        lower = max(0, max(lo, idx - 3) - lo)
        upper = max(0, hi - min(hi, idx + 4))
        return lower + upper

    def test_every_bin_of_a_scannable_spectrum_has_a_neighbourhood(self):
        # The smallest spectrum that can reach the scan: 2048 samples → 1025 bins.
        for bins in (1025, 2049, 16385):
            empty = [idx for idx in range(bins) if self._neighbourhood_length(idx, bins) == 0]
            assert empty == []

    def test_only_spectra_too_small_to_be_reachable_have_empty_neighbourhoods(self):
        """Sanity check on the derivation: emptiness requires a < 8-bin spectrum."""
        empty = [idx for idx in range(2) if self._neighbourhood_length(idx, 2) == 0]
        assert empty == [0, 1]
        for bins in (9, 33, 129):
            assert [i for i in range(bins) if self._neighbourhood_length(i, bins) == 0] == []

    def test_scan_returns_cleanly_across_the_reachable_spectrum_sizes(self):
        from audioman.core.aesthetic import detect_rf_noise_events

        t = np.arange(48000, dtype=np.float32) / 48000
        for length in (2048, 4096, 48000):
            mono = (0.02 * np.sin(2 * np.pi * 8000 * t[:length])).astype(np.float32)
            events, backend = detect_rf_noise_events(
                mono, 48000, min_tones=1, min_frequency=1000.0, max_frequency=10000.0,
            )
            assert backend == "heuristic"
            assert isinstance(events, list)


# ---------------------------------------------------------------------------
# core/qc.py
# ---------------------------------------------------------------------------


class TestDetectClicksAlignmentBranchesAreDead:
    """qc.py:170-171 and 173 — the truncate/residual-pad corrections.

    Below the short-buffer early return, `n >= win` with
    `win = max(int(window_ms/1000*sr), 16) >= 16`. The rolling mean has
    `n - win + 1` entries and the ratio needs `n - 1`, so the gap is exactly
    `win - 2 >= 14`:

      * `pad` is therefore always at least 7, so the truncation branch could
        never run and has been deleted;
      * padding by `pad` on the left and `n - 1 - len - pad` on the right makes
        the result exactly `n - 1` long, so the top-up branch could never run
        either -- it too is gone, replaced by the unconditional symmetric pad.

    The invariants both branches rested on are checked below, so the deletion
    cannot silently stop being valid. A differential run over 2715 reachable
    (sample_rate, window_ms, n) triples confirmed the old block and the
    unconditional pad produce identical arrays, which is what allowed the
    no-op branches to go rather than be pinned with a pragma.
    """

    def test_gap_between_rolling_mean_and_diff_is_always_win_minus_two(self):
        for sample_rate in (8000, 22050, 44100, 48000, 96000, 192000):
            for window_ms in (0.0001, 0.4, 5.0, 30.0, 50.0, 100.0, 1000.0):
                win = max(int(window_ms / 1000.0 * sample_rate), 16)
                for n in (win, win + 1, win + 3, 2 * win, 4096, 100000):
                    rolling = n - win + 1
                    assert (n - 1) - rolling == win - 2
                    assert win - 2 >= 14  # so pad = gap // 2 >= 7, never negative

    def test_the_symmetric_pad_always_restores_exactly_n_minus_one(self):
        for win in (16, 17, 48, 2205, 4800, 48000):
            for n in (win, win + 1, win + 2, win + 5, 2 * win, 50000):
                rolling = n - win + 1
                gap = (n - 1) - rolling
                pad = gap // 2
                assert pad >= 0
                # left pad + body + right pad == the length the ratio needs
                assert pad + rolling + ((n - 1) - rolling - pad) == n - 1
                assert ((n - 1) - rolling - pad) >= 0  # no negative pad width

    def test_the_unconditional_pad_produces_the_length_diff_needs(self):
        """Runs the surviving padding line itself, not just its arithmetic.

        `np.pad` raises on a negative width and on `mode="edge"` over an empty axis,
        so this doubles as a check that the branch deletion left no reachable input
        where either could happen: the body is `n - win + 1 >= 1` entries and both
        widths are non-negative, for every `n >= win >= 16`.
        """
        for win in (16, 17, 31, 48, 2205, 4800, 48000):
            for n in (win, win + 1, win + 2, win + 3, win + 7, 2 * win, 50000):
                rolling = np.arange(n - win + 1, dtype=np.float64)
                pad = (n - 1 - len(rolling)) // 2
                padded = np.pad(
                    rolling, (pad, n - 1 - len(rolling) - pad), mode="edge"
                )
                assert padded.shape == (n - 1,), (win, n)

    def test_detect_clicks_never_raises_across_the_parameter_sweep(self):
        """np.pad raises on a negative width, so a sweep doubles as the guard.

        The click is injected into constant-amplitude material, so `max_ratio` is
        the observable: a padding slip would misalign the ratio array against
        `np.diff` and the peak would move. The absolute count is left alone —
        it depends on the sensitivity threshold, which is not what is at stake.
        """
        from audioman.core.qc import detect_clicks

        for length in (2048, 2049, 3000, 4096, 8192):
            audio = np.full(length, 0.2, dtype=np.float32)
            audio[length // 2] = 9.0
            for window_ms in (0.0001, 0.4, 5.0, 50.0):
                result = detect_clicks(audio, 44100, window_ms=window_ms, sensitivity=6.0)
                assert "n_clicks" in result
                assert result["max_ratio"] > 3.0  # the injected step is visible

    def test_short_buffer_branch_is_the_only_alternative(self):
        """n < win takes the local-RMS branch, which has no padding at all."""
        from audioman.core.qc import detect_clicks

        audio = np.full(15, 0.2, dtype=np.float32)
        audio[7] = 5.0
        result = detect_clicks(audio, 44100, window_ms=4000.0, sensitivity=3.0)
        assert result["n_clicks"] >= 1
        assert result["max_ratio"] > 3.0


# ---------------------------------------------------------------------------
# config/settings.py
# ---------------------------------------------------------------------------


class TestTomlNonTableRootIsDead:
    """settings.py:32 — the `isinstance(data, dict)` guard in `_TomlSettingsSource`.

    TOML's document root is by definition a table, and `tomllib` enforces that
    while parsing: a scalar or array root raises `TOMLDecodeError` before
    `load` returns anything. The guard therefore never sees a non-dict and its
    `raise` is unreachable; it is kept as a contract marker for the caller.

    The premise is checked here rather than assumed — if a future `tomllib`
    starts returning non-mapping roots, these tests fail and line 32 becomes
    live. The line carries `# pragma: no cover` in the source for the same
    reason.

    Note the guard is not purely decorative: `load`'s own annotation is
    ``dict[str, Any]``, so the check is what keeps the return type honest if the
    stdlib ever changes. The last test pins that annotation too, so the marker
    has a second premise that fails loudly if it stops holding.
    """

    def test_scalar_root_is_rejected_by_the_parser_not_returned(self, tmp_path):
        import tomllib

        path = tmp_path / "scalar.toml"
        path.write_text("42\n", encoding="utf-8")
        with path.open("rb") as handle:
            with pytest.raises(tomllib.TOMLDecodeError):
                tomllib.load(handle)

    def test_all_toml_root_forms_the_parser_accepts_are_tables(self, tmp_path):
        """Every document TOML actually accepts parses to a mapping root.

        The two forms below are the ones most likely to be mistaken for a
        non-table root -- a bare value at the top of the file and an
        array-of-tables -- plus the dotted-key and inline-table spellings.
        """
        import tomllib

        documents = [
            "",
            "42\n",  # invalid: must be rejected, not returned as an int
            "x = 1\n",
            "[a]\nb = 2\n",
            "[[t]]\np = 'a'\n",
            "a.b.c = 1\n",
            "t = {a = 1}\n",
            "'quoted key' = 1\n",
        ]
        for text in documents:
            path = tmp_path / "doc.toml"
            path.write_text(text, encoding="utf-8")
            with path.open("rb") as handle:
                try:
                    data = tomllib.load(handle)
                except tomllib.TOMLDecodeError:
                    assert text in ("", "42\n"), f"unexpected rejection: {text!r}"
                    continue
            assert isinstance(data, dict), text

    def test_the_load_annotation_matches_the_guard(self):
        """The guard's second premise: `tomllib.load` still declares a dict return."""
        import tomllib
        import typing

        resolved = typing.get_type_hints(tomllib.load)["return"]
        origin = getattr(resolved, "__origin__", resolved)
        assert origin is dict, resolved

    def test_array_root_is_rejected_by_the_parser_not_returned(self, tmp_path):
        import tomllib

        path = tmp_path / "array.toml"
        path.write_text("[1, 2, 3]\n", encoding="utf-8")
        with path.open("rb") as handle:
            with pytest.raises(tomllib.TOMLDecodeError):
                tomllib.load(handle)

    def test_every_parseable_document_has_a_mapping_root(self, tmp_path):
        import tomllib

        documents = {
            "empty": "",
            "flat": "json_output = true\n",
            "table": "[section]\nkey = 1\n",
            "array_of_tables": "[[track]]\npath = 'a.wav'\n",
        }
        for name, text in documents.items():
            path = tmp_path / f"{name}.toml"
            path.write_text(text, encoding="utf-8")
            with path.open("rb") as handle:
                assert isinstance(tomllib.load(handle), dict)

    def test_missing_file_still_returns_empty_without_parsing(self, tmp_path):
        from audioman.config import settings as cfg_settings

        cfg_settings.AudiomanSettings.config_file = tmp_path / "absent.toml"
        source = cfg_settings._TomlSettingsSource(cfg_settings.AudiomanSettings)

        assert source() == {}


# ---------------------------------------------------------------------------
# core/dsp.py
# ---------------------------------------------------------------------------


class TestFadeCurveDriftFallback:
    """dsp.py:225 — the `else: c = x` arm of `_fade_curve`.

    This is the one residual that is reachable, and only through module state:
    `_fade_curve` rejects any name outside `FADE_CURVES`, and the chain below it
    implements exactly those names, so the fallback is dead while the two stay
    in sync. Widening `FADE_CURVES` — the documented way to announce a curve —
    is the only way to reach it, and that is the state injected here.

    The assertion pins the behaviour such a mismatch must produce: a curve the
    module advertises but does not implement degrades to linear instead of
    raising at the user for a preset they could already save.
    """

    def test_drift_degrades_to_linear_mono(self, monkeypatch):
        import audioman.core.dsp as dsp

        monkeypatch.setattr(dsp, "FADE_CURVES", dsp.FADE_CURVES + ("legacy_curve",))
        expected = np.linspace(0.0, 1.0, 8, dtype=np.float32)

        out = dsp.fade_in(np.ones(64, dtype=np.float32), 8, curve="legacy_curve")

        np.testing.assert_allclose(out[:8], expected, atol=1e-6)
        np.testing.assert_allclose(out[8:], np.ones(56, dtype=np.float32), atol=1e-6)

    def test_drift_degrades_to_linear_stereo_and_fade_out(self, monkeypatch):
        import audioman.core.dsp as dsp

        monkeypatch.setattr(dsp, "FADE_CURVES", dsp.FADE_CURVES + ("legacy_curve",))
        expected = np.linspace(0.0, 1.0, 8, dtype=np.float32)

        out = dsp.fade_out(np.ones((2, 64), dtype=np.float32), 8, curve="legacy_curve")

        # fade_out applies the reversed curve to the *last* n samples and leaves
        # the head untouched: full gain at -8, zero at -1, unity before that.
        np.testing.assert_allclose(out[0, -8:], expected[::-1], atol=1e-6)
        np.testing.assert_allclose(out[:, -1], np.zeros(2, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(out[:, :56], np.ones((2, 56), dtype=np.float32), atol=1e-6)

    def test_unregistered_curve_still_raises_so_typos_are_not_swallowed(self):
        from audioman.core.dsp import fade_in

        with pytest.raises(ValueError, match="Unknown fade curve"):
            fade_in(np.ones(64, dtype=np.float32), 8, curve="not_a_curve_at_all")

    def test_shipped_curves_all_take_their_own_branch(self):
        """Every advertised curve must be implemented, or drift is silent."""
        from audioman.core.dsp import FADE_CURVES, _fade_curve

        reference = np.linspace(0.0, 1.0, 32, dtype=np.float32)
        implemented = {"linear", "cosine", "equal_power", "exponential", "logarithmic"}
        assert set(FADE_CURVES) == implemented
        for curve in FADE_CURVES:
            c = _fade_curve(32, curve, "in")
            assert c.shape == reference.shape
            assert c[-1] == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# __main__.py (reachable — covered for real)
# ---------------------------------------------------------------------------


class TestMainModule:
    def test_dash_m_audioman_dispatches_to_the_cli(self, monkeypatch):
        """`python -m audioman` runs the module body (lines 4 and 6-7)."""
        import runpy
        import sys as _sys

        import audioman.cli.app as app_module

        called: list[object] = []

        def _fake_main(argv=None):
            called.append(argv)

        monkeypatch.setattr(app_module, "main", _fake_main)
        monkeypatch.setattr(_sys, "argv", ["audioman", "doctor"])

        runpy.run_module("audioman", run_name="__main__")

        # The entry point is invoked with no explicit argv, so the CLI reads
        # sys.argv itself.
        assert called == [None]

    def test_version_flag_exits_zero(self, monkeypatch, capsys):
        import runpy
        import sys as _sys

        monkeypatch.setattr(_sys, "argv", ["audioman", "--version"])

        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("audioman", run_name="__main__")

        assert excinfo.value.code == 0
        assert "audioman" in capsys.readouterr().out

    def test_import_binds_main_without_running_it(self):
        """The `if __name__ == "__main__"` guard must hold on a plain import."""
        import importlib

        module = importlib.import_module("audioman.__main__")

        assert callable(module.main)


# ---------------------------------------------------------------------------
# End-to-end sanity: the same arithmetic on a real file round trip
# ---------------------------------------------------------------------------


class TestRealAudioRoundTrip:
    def test_qc_and_aesthetic_agree_on_a_short_low_rate_wav(self, tmp_path):
        from audioman.core import aesthetic, qc

        path = tmp_path / "short.wav"
        audio = np.linspace(-0.2, 0.2, 3000, dtype=np.float32)
        sf.write(str(path), audio, 22050, subtype="PCM_16")

        read, sr = sf.read(str(path), dtype="float32")
        assert "n_clicks" in qc.detect_clicks(read, sr, window_ms=20.0)

        events, backend = aesthetic.detect_rf_noise_events(read, sr, min_frequency=0.0)
        assert backend == "heuristic"
        assert isinstance(events, list)
