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
# and weighted by how far the primary cue is beyond its floor. When the cues against
# a term outweigh or cancel the cues for it (net agreement <= 0) the term is not
# proposed: it is listed under "contradicted" with its evidence instead, so the
# words an agent reads first are never ones the measurements argue against.
#
# Significant is not the same as audible: on a long steady file a 7 Hz centroid
# shift can be many floors wide. Each rule therefore also names a provisional
# perceptual minimum for its primary cue (relative to A, or absolute in the cue's
# unit); a primary change below it yields no term, and the rule is listed under
# "below_perceptual_minimum" with its evidence. The minimums are uncalibrated
# placeholders until listening (ABX) data replaces them.

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
    # smallest primary change that earns the word: ("rel", fraction of A) or ("abs", amount in the cue's unit)
    min_delta: tuple[str, float] | None = None


_BRIGHT = ("rel", 0.03)
_FLAT = ("abs", 1.0)
_WIDTH = ("abs", 1.0)
_LEVEL = ("abs", 0.5)
_DYN = ("abs", 1.0)
_ONSET = ("abs", 0.5)
_PITCH = ("abs", 10.0)
_INTONATION = ("abs", 5.0)


# Descriptor names are texture descriptors; "band:<name>" refers to a named band.
RULES = (
    Rule("darker", "brighter", ("spectral_centroid", -1),
         (("hf_ratio", -1), ("sharpness", -1), ("spectral_rolloff", -1), ("band:air", -1)), _BRIGHT),
    Rule("brighter", "darker", ("spectral_centroid", +1),
         (("hf_ratio", +1), ("sharpness", +1), ("spectral_rolloff", +1), ("band:air", +1)), _BRIGHT),
    Rule("noisier", "cleaner", ("spectral_flatness", +1),
         (("spectral_flux", +1), ("spectral_contrast", -1)), _FLAT),
    Rule("cleaner", "noisier", ("spectral_flatness", -1),
         (("spectral_flux", -1), ("spectral_contrast", +1)), _FLAT),
    Rule("wider", "narrower", ("stereo_width", +1), (("lr_correlation", -1),), _WIDTH),
    Rule("narrower", "wider", ("stereo_width", -1), (("lr_correlation", +1),), _WIDTH),
    Rule("louder", "quieter", ("rms", +1), (), _LEVEL),
    Rule("quieter", "louder", ("rms", -1), (), _LEVEL),
    Rule("more compressed", "more dynamic", ("dynamic_range", -1), (("crest_factor", -1),), _DYN),
    Rule("more dynamic", "more compressed", ("dynamic_range", +1), (("crest_factor", +1),), _DYN),
    Rule("more transient", "smoother", ("onset_density", +1), (("spectral_flux", +1),), _ONSET),
    Rule("smoother", "more transient", ("onset_density", -1), (("spectral_flux", -1),), _ONSET),
    Rule("harmonically richer", "purer", ("spectral_bandwidth", +1),
         (("crest_factor", -1), ("hf_ratio", +1), ("spectral_contrast", +1))),
    Rule("pitch-corrected", "looser pitch", ("pitch_deviation", -1), (("pitch_spread", -1),), _INTONATION),
    Rule("looser pitch", "pitch-corrected", ("pitch_deviation", +1), (("pitch_spread", +1),), _INTONATION),
    Rule("pitched up", "pitched down", ("pitch", +1), (), _PITCH),
    Rule("pitched down", "pitched up", ("pitch", -1), (), _PITCH),
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
    return {"delta": d["delta"], "significant": d["significant"], "floor": d.get("floor"), "unit": d.get("unit", ""),
            "a": d.get("a")}


def _perceptible(rule: Rule, primary: dict) -> bool:
    if rule.min_delta is None:
        return True
    kind, amount = rule.min_delta
    if kind == "rel":
        return primary["a"] is not None and abs(primary["delta"]) >= amount * abs(primary["a"])
    return abs(primary["delta"]) >= amount


def _agreement(cue: dict | None, direction: int) -> int:
    if cue is None or not cue["significant"]:
        return 0
    return 1 if (cue["delta"] > 0) == (direction > 0) else -1


def propose_terms(diff: dict) -> dict[str, Any]:
    """Terms for a diff_audio() result, strongest first, each with evidence and confidence."""
    terms, contradicted, too_small = [], [], []
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
        entry = {"term": rule.term, "opposite": rule.opposite, "confidence": round(confidence, 3),
                 "evidence": evidence}
        if not _perceptible(rule, primary):
            kind, amount = rule.min_delta
            too_small.append({"term": rule.term, "opposite": rule.opposite,
                              "minimum": f"{amount:.0%} of A" if kind == "rel" else f"{amount:g} {primary['unit']}",
                              "evidence": evidence})
        elif sum(agreements) <= 0:
            contradicted.append({k: v for k, v in entry.items() if k != "confidence"})
        else:
            terms.append(entry)
    terms.sort(key=lambda t: -t["confidence"])
    return {"note": EXPERIMENTAL_NOTE, "terms": terms, "contradicted": contradicted,
            "below_perceptual_minimum": too_small}
