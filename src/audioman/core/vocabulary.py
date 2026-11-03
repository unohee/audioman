# Created: 2026-11-04
# Purpose: EXPERIMENTAL mapping from significant descriptor changes (a diff) to plain
#          words (darker, noisier, wider, ...), each with its evidence and a confidence.
# Dependencies: stdlib only
#
# Words are hypotheses, not verdicts. A term is proposed only from changes the diff
# already judged significant against their floors, it always lists the measured
# evidence behind it, and its confidence says how much of that evidence agrees.
# The mapping is uncalibrated against listeners: treat a term as "the numbers moved
# the way this word usually describes", and check the evidence before repeating it.
#
# Rules: each term names cue descriptors with the direction that supports it. A
# term is proposed when its primary cue is significant in that direction;
# confidence = mean over cues of agreement (1 if significant in the supporting
# direction, 0 if not significant, -1 if significant against), clipped to [0, 1]
# and weighted by how far the primary cue is beyond its floor.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

EXPERIMENTAL_NOTE = ("experimental: descriptor-to-word mapping, not calibrated against listeners; "
                     "read the evidence before relying on a term")


@dataclass(frozen=True)
class Rule:
    term: str
    opposite: str
    primary: tuple[str, int]  # (descriptor or band, +1 up / -1 down) that must be significant
    support: tuple[tuple[str, int], ...] = ()


# Descriptor names are texture descriptors; "band:<name>" refers to a named band.
RULES = (
    Rule("darker", "brighter", ("spectral_centroid", -1),
         (("hf_ratio", -1), ("sharpness", -1), ("spectral_rolloff", -1), ("band:air", -1))),
    Rule("brighter", "darker", ("spectral_centroid", +1),
         (("hf_ratio", +1), ("sharpness", +1), ("spectral_rolloff", +1), ("band:air", +1))),
    Rule("noisier", "cleaner", ("spectral_flatness", +1),
         (("spectral_flux", +1), ("spectral_contrast", -1))),
    Rule("cleaner", "noisier", ("spectral_flatness", -1),
         (("spectral_flux", -1), ("spectral_contrast", +1))),
    Rule("wider", "narrower", ("stereo_width", +1), (("lr_correlation", -1),)),
    Rule("narrower", "wider", ("stereo_width", -1), (("lr_correlation", +1),)),
    Rule("louder", "quieter", ("rms", +1), ()),
    Rule("quieter", "louder", ("rms", -1), ()),
    Rule("more compressed", "more dynamic", ("dynamic_range", -1), (("crest_factor", -1),)),
    Rule("more dynamic", "more compressed", ("dynamic_range", +1), (("crest_factor", +1),)),
    Rule("more transient", "smoother", ("onset_density", +1), (("spectral_flux", +1),)),
    Rule("smoother", "more transient", ("onset_density", -1), (("spectral_flux", -1),)),
    Rule("harmonically richer", "purer", ("spectral_bandwidth", +1),
         (("crest_factor", -1), ("hf_ratio", +1), ("spectral_contrast", +1))),
)


def _cue(diff: dict, cue: str) -> dict | None:
    """Normalise a descriptor or named band of a diff result to {delta, significant, floor, unit}."""
    if cue.startswith("band:"):
        name = cue.split(":", 1)[1]
        for band in diff.get("named_bands") or []:
            if band["band"] == name:
                return {"delta": band["delta_db"], "significant": band["significant"],
                        "floor": band["floor_db"], "unit": "dB"}
        return None
    d = (diff.get("descriptors") or {}).get(cue)
    if not d or d.get("delta") is None:
        return None
    return {"delta": d["delta"], "significant": d["significant"], "floor": d.get("floor"), "unit": d.get("unit", "")}


def _agreement(cue: dict | None, direction: int) -> int:
    if cue is None or not cue["significant"]:
        return 0
    return 1 if (cue["delta"] > 0) == (direction > 0) else -1


def propose_terms(diff: dict) -> dict[str, Any]:
    """Terms for a diff_audio() result, strongest first, each with evidence and confidence."""
    terms = []
    for rule in RULES:
        primary = _cue(diff, rule.primary[0])
        if _agreement(primary, rule.primary[1]) != 1:
            continue
        cues = [rule.primary, *rule.support]
        agreements = [_agreement(_cue(diff, name), sign) for name, sign in cues]
        floor = primary["floor"] or 0.0
        strength = abs(primary["delta"]) / floor if floor else 100.0
        confidence = max(0.0, min(1.0, sum(agreements) / len(agreements))) * min(1.0, strength / 10.0)
        evidence = []
        for (name, sign), agree in zip(cues, agreements):
            cue = _cue(diff, name)
            if cue is not None:
                evidence.append({"cue": name, "delta": cue["delta"], "unit": cue["unit"], "floor": cue["floor"],
                                 "supports": agree == 1, "contradicts": agree == -1})
        terms.append({"term": rule.term, "opposite": rule.opposite, "confidence": round(confidence, 3),
                      "evidence": evidence})
    terms.sort(key=lambda t: -t["confidence"])
    return {"note": EXPERIMENTAL_NOTE, "terms": terms}
