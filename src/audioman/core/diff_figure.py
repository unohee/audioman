# Created: 2026-11-04
# Purpose: PNG figure for `audioman diff --png`: per-ERB-band level change with its floor,
#          and the whole-file descriptor changes in units of their floor.
# Dependencies: matplotlib (Agg)

from __future__ import annotations

from pathlib import Path
from typing import Any


_STYLE = {"significant": ("tab:red", ""), "small": ("tab:orange", "//"), "noise": ("tab:gray", "")}


def _state(d: dict[str, Any]) -> str:
    """significant, small (clears the floor test but not the per-unit minimum) or noise."""
    if d["significant"]:
        return "significant"
    if d["delta"] is not None and abs(d["delta"]) > 2 * (d["floor"] or 0.0):
        return "small"
    return "noise"


def render(result: dict[str, Any], path: str | Path, *, title: str = "B - A") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = [b for b in result["erb_bands"] if b["delta"] is not None]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

    centers = [b["center_hz"] for b in bands]
    deltas = [b["delta"] for b in bands]
    floors = [2 * b["floor"] for b in bands]  # the significance band (k = 2)
    colors = ["tab:red" if b["significant"] else "tab:gray" for b in bands]
    ax1.fill_between(centers, [-f for f in floors], floors, color="0.85", label="2 x floor (not significant)")
    ax1.scatter(centers, deltas, c=colors, s=18, zorder=3)
    ax1.plot(centers, deltas, color="0.4", lw=0.8)
    ax1.axhline(0, color="black", lw=0.6)
    ax1.set_xscale("log")
    ax1.set_xlabel("ERB band centre (Hz)")
    ax1.set_ylabel("level change (dB)")
    ax1.set_title(f"{title}: per-band level change (red = significant)")
    ax1.legend(loc="lower left", fontsize=8)

    names = list(result["descriptors"])
    ratios = []
    for name in names:
        d = result["descriptors"][name]
        if d["delta"] is None:
            ratios.append(0.0)
        elif d["floor"]:
            ratios.append(d["delta"] / d["floor"])
        else:
            ratios.append(0.0 if d["delta"] == 0 else (1e3 if d["delta"] > 0 else -1e3))
    states = [_state(result["descriptors"][n]) for n in names]
    from matplotlib.patches import Patch

    bars = ax2.barh(names, [max(-50.0, min(50.0, r)) for r in ratios], color=[_STYLE[s][0] for s in states])
    for bar, state in zip(bars, states):
        bar.set_hatch(_STYLE[state][1])
        bar.set_edgecolor("white" if state == "small" else bar.get_facecolor())
    ax2.axvline(0, color="black", lw=0.6)
    ax2.legend(handles=[Patch(facecolor=_STYLE[k][0], hatch=_STYLE[k][1], edgecolor="white", label=label)
                        for k, label in (("significant", "significant"),
                                         ("small", "beyond 2 x floor but below the per-unit minimum"),
                                         ("noise", "within 2 x floor"))],
               loc="lower right", fontsize=8)
    ax2.set_xlabel("change / floor (clipped at +/-50); significant = beyond max(2 x floor, per-unit minimum)")
    ax2.set_title("descriptor changes in units of their repeat floor")

    out = Path(path)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
