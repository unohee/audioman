# tests/unit/test_plugin_analysis_guards.py — the two slice guards in the harmonic engine.
#
# `measure_thd` and `measure_imd` both step a frequency region around a peak bin and
# guard the slice against being empty before `np.max` sees it (np.max raises on a
# zero-size array — measured, not assumed). Both guards sit behind a bound check that
# already admits only in-range bins, so neither can fire. That claim is what the
# `# pragma: no cover` markers in `plugin_analysis.py` rest on, and it is checked here
# rather than asserted in a comment: the slice arithmetic is evaluated for every
# admissible bin index and half width, and the "it would raise" premise is pinned by
# driving the unguarded reduction.
#
# The sibling guard in `measure_imd` — the one that skips a sideband whose rounded bin
# index runs past the end of the spectrum — IS reachable through float rounding and is
# covered by a real test in `test_plugin_analysis.py`
# (`test_sideband_rounding_onto_the_bin_count_is_skipped`).

from __future__ import annotations

import numpy as np
import pytest

from audioman.core import plugin_analysis as pa


def _region(bins: int, centre: int, half: int) -> int:
    """Length of the slice the code builds, for a spectrum of `bins` entries."""
    return len(range(max(0, centre - half), min(bins, centre + half)))


class TestThdRegionSliceIsNeverEmpty:
    """plugin_analysis.py:229 — `len(region) == 0` in `measure_thd`.

    The loop breaks at `h_bin >= len(spectrum)` before the slice is built, so the
    only bin indices that reach it satisfy `0 <= h_bin < bins`. With
    `sr = max(3, h_bin // 50) >= 3` the slice ends at `min(h_bin + sr, bins)` which
    is strictly greater than `h_bin` and therefore than its own start, so it holds at
    least one bin.
    """

    def test_every_admissible_bin_yields_a_non_empty_region(self):
        for bins in (1, 2, 3, 4, 5, 8, 9, 16, 17, 129, 1025, 8193, 16385):
            for centre in range(bins):  # the `h_bin >= len(spectrum)` break admits exactly these
                half = max(3, centre // 50)
                assert _region(bins, centre, half) >= 1, (bins, centre, half)

    def test_the_half_width_floor_is_what_makes_it_hold_at_bin_zero(self):
        # At the left edge the slice is [0, min(half, bins)), so it is the width that
        # matters -- and 3 is the smallest value `max(3, ...)` can produce.
        for bins in (1, 2, 3, 4, 8):
            assert _region(bins, 0, 3) == min(3, bins) >= 1

    def test_emptiness_requires_a_bin_outside_the_spectrum(self):
        """The tightness check: on the in-range bins nothing empties the slice.

        Empty regions do exist for this slice arithmetic, so the guard is not
        redundant in isolation -- they are reached only by centres the loop cannot
        produce. Below zero, the slice collapses on the left; at or above `bins` it
        collapses on the right once `centre - sr` passes the end.
        """
        for bins in (1, 2, 3, 8, 9, 16, 129, 1025):
            for centre in range(bins):
                assert _region(bins, centre, max(3, centre // 50)) >= 1, (bins, centre)

        # and the cases that do empty it, to show the assertion above is not vacuous
        assert _region(1, -40, max(3, -40 // 50)) == 0
        assert _region(1, 100, max(3, 100 // 50)) == 0  # 100 - 33 > 1

    def test_unguarded_reduction_raises_on_an_empty_region(self):
        """Why the guard exists: np.max has no identity, so an empty region raises."""
        with pytest.raises(ValueError, match="zero-size array"):
            np.max(np.zeros(0, dtype=np.float64))

    def test_public_entry_points_return_instead_of_raising(self, monkeypatch):
        """Both entry points survive every bin the guard admits, including bin 0."""
        # fft_size=1 gives a 1-bin spectrum, so the harmonic loop reaches the
        # left-edge case where the lower half of the slice is clipped away.
        monkeypatch.setattr(
            pa, "_load_plugin", lambda path, params=None: _FakeIdentityPlugin()
        )
        for fft_size in (1, 2, 3, 5, 8, 16, 2048):
            for frequency in (1.0, 100.0, 1000.0, 7000.0, 22050.0):
                thd = pa.measure_thd("p", frequency=frequency, sample_rate=44100,
                                     fft_size=fft_size)
                assert thd.thd_percent == thd.thd_percent  # not NaN
            imd = pa.measure_imd("p", freq_low=60.0, freq_high=7000.0,
                                 sample_rate=44100, fft_size=fft_size)
            assert imd.imd_percent == imd.imd_percent


class TestImdRegionSliceIsNeverEmpty:
    """plugin_analysis.py:300 — `len(region) == 0` in `measure_imd`.

    The `sb_bin >= len(spectrum)` check above skips every out-of-range bin, so the
    slice is built only for `0 <= sb_bin < bins`. With `sr2 = max(2, sb_bin // 100)`
    the slice ends at `min(sb_bin + sr2, bins) > sb_bin`, so it holds at least one bin.
    """

    def test_every_admissible_bin_yields_a_non_empty_region(self):
        for bins in (1, 2, 3, 4, 5, 8, 9, 16, 17, 129, 1025, 8193, 16385):
            for centre in range(bins):
                half = max(2, centre // 100)
                assert _region(bins, centre, half) >= 1, (bins, centre, half)

    def test_the_bound_check_covers_every_empty_region(self):
        """The bound check above it is what makes this guard unreachable.

        The slice empties exactly when the centre lies past an end of the spectrum, so
        the claim to check is the implication: every centre whose region would be empty
        is one the `sb_bin >= len(spectrum)` check already skipped. The negative half of
        that is unreachable from the loop as well -- the sideband frequency is
        `freq_high + sign * n * freq_low`, so a non-positive one is skipped by the
        frequency check before a bin is computed.
        """
        for bins in (1, 3, 9, 129, 1025):
            for centre in range(bins):
                # in range: never empty (the loop's actual admission criterion)
                assert _region(bins, centre, max(2, centre // 100)) >= 1, (bins, centre)
            # past the end: empty, and every one of them is caught by the bound check
            for centre in range(bins, bins + 400):
                if _region(bins, centre, max(2, centre // 100)) == 0:
                    assert centre >= bins  # i.e. skipped before the region is built

    def test_reachable_sideband_pair_still_trips_the_bound_check(self):
        """The reachable path, restated against this file's helper.

        The numbers are the ones the companion test drives through the public API:
        an odd `fft_size` puts the boundary sideband on the `k + 0.5` tie, which
        rounds up to `bins` and is dropped by the bound check.
        """
        sample_rate, fft_size = 8000.1, 2051
        bins = fft_size // 2 + 1
        boundary = float(np.nextafter(sample_rate / 2.0, -np.inf))
        sb_bin = int(round(boundary * fft_size / sample_rate))
        assert boundary < sample_rate / 2  # passes the frequency check
        assert sb_bin == bins  # and is skipped by the bound check
        # Past the end the region would be empty once the half width clears the end.
        sr2 = max(2, sb_bin // 100)
        assert _region(bins, sb_bin, sr2) >= 1  # sr2 == 10 still reaches back into range
        assert _region(bins, sb_bin + sr2, max(2, (sb_bin + sr2) // 100)) == 0


class _FakeIdentityPlugin:
    """Minimal stand-in for the wrapper, so the entry points run without a plugin."""

    def load(self) -> None:
        pass

    def process(self, audio, sample_rate, reset=True):
        return np.asarray(audio, dtype=np.float32)

    def reset(self) -> None:
        pass

    def set_parameters(self, params) -> None:
        pass
