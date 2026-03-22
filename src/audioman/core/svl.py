# Created: 2026-03-22
# Purpose: Sonic Visualiser .svl 레이어 파일 생성
# Dependencies: xml.etree (stdlib)

from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent

import numpy as np


def write_time_instants(
    path: Path,
    frames: list[int],
    labels: list[str] | None = None,
    sample_rate: int = 44100,
    resolution: int = 512,
) -> None:
    """Time Instants 레이어 (온셋, 비트 등)"""
    if labels is None:
        labels = [""] * len(frames)

    sv = Element("sv")
    data = SubElement(sv, "data")

    SubElement(data, "model",
        id="0", name="instants",
        sampleRate=str(sample_rate),
        start="0", end=str(max(frames) if frames else 0),
        type="sparse", dimensions="1",
        resolution=str(resolution),
        notifyOnAdd="true", dataset="0",
    )

    dataset = SubElement(data, "dataset", id="0", dimensions="1")
    for frame, label in zip(frames, labels):
        SubElement(dataset, "point", frame=str(int(frame)), label=str(label))

    _write_xml(sv, path)


def write_time_values(
    path: Path,
    frames: list[int],
    values: list[float],
    units: str = "",
    name: str = "analysis",
    sample_rate: int = 44100,
    resolution: int = 512,
) -> None:
    """Time Values 레이어 (spectral centroid, 에너지 등)"""
    sv = Element("sv")
    data = SubElement(sv, "data")

    SubElement(data, "model",
        id="0", name=name,
        sampleRate=str(sample_rate),
        start="0", end=str(max(frames) if frames else 0),
        type="sparse", dimensions="2",
        resolution=str(resolution),
        notifyOnAdd="true", dataset="0",
        minimum=str(float(min(values))) if values else "0",
        maximum=str(float(max(values))) if values else "0",
        units=units,
    )

    dataset = SubElement(data, "dataset", id="0", dimensions="2")
    for frame, value in zip(frames, values):
        SubElement(dataset, "point",
            frame=str(int(frame)),
            value=str(float(value)),
            label="",
        )

    _write_xml(sv, path)


def write_notes(
    path: Path,
    frames: list[int],
    pitches: list[float],
    durations: list[int],
    levels: list[float] | None = None,
    labels: list[str] | None = None,
    units: str = "MIDI Pitch",
    sample_rate: int = 44100,
    resolution: int = 512,
) -> None:
    """Notes 레이어 (피치 추정, MIDI 노트 등)"""
    if levels is None:
        levels = [1.0] * len(frames)
    if labels is None:
        labels = [""] * len(frames)

    sv = Element("sv")
    data = SubElement(sv, "data")

    SubElement(data, "model",
        id="0", name="notes",
        sampleRate=str(sample_rate),
        start="0",
        end=str(max(f + d for f, d in zip(frames, durations)) if frames else 0),
        type="sparse", dimensions="3", subtype="note",
        resolution=str(resolution),
        notifyOnAdd="true", dataset="0",
        minimum=str(float(min(pitches))) if pitches else "0",
        maximum=str(float(max(pitches))) if pitches else "0",
        units=units,
    )

    dataset = SubElement(data, "dataset", id="0", dimensions="3")
    for frame, pitch, dur, level, label in zip(frames, pitches, durations, levels, labels):
        SubElement(dataset, "point",
            frame=str(int(frame)),
            value=str(float(pitch)),
            duration=str(int(dur)),
            level=str(float(level)),
            label=str(label),
        )

    _write_xml(sv, path)


def write_dense3d(
    path: Path,
    matrix: np.ndarray,
    sample_rate: int = 44100,
    window_size: int = 1024,
    hop_size: int = 512,
    bin_names: list[str] | None = None,
) -> None:
    """Dense 3D 레이어 (스펙트로그램, 크로마그램 등)

    Args:
        matrix: (n_frames, n_bins) 배열
        window_size: FFT 윈도우 크기
        hop_size: 홉 크기 (resolution)
        bin_names: 각 bin의 이름 (예: "0-86Hz")
    """
    n_frames, n_bins = matrix.shape

    sv = Element("sv")
    data = SubElement(sv, "data")

    SubElement(data, "model",
        id="0", name="spectrogram",
        sampleRate=str(sample_rate),
        start="0",
        type="dense", dimensions="3",
        windowSize=str(window_size),
        yBinCount=str(n_bins),
        minimum=str(float(matrix.min())),
        maximum=str(float(matrix.max())),
        startFrame="0",
        dataset="0",
    )

    dataset = SubElement(data, "dataset", id="0", dimensions="3", separator=" ")

    if bin_names:
        for i, name in enumerate(bin_names):
            SubElement(dataset, "bin", number=str(i), name=name)

    for i in range(n_frames):
        row = SubElement(dataset, "row", n=str(i))
        row.text = " ".join(f"{v:.6f}" for v in matrix[i])

    _write_xml(sv, path)


def _write_xml(root: Element, path: Path) -> None:
    """XML 파일 작성"""
    tree = ElementTree(root)
    indent(tree, space="  ")
    tree.write(str(path), xml_declaration=True, encoding="UTF-8")
