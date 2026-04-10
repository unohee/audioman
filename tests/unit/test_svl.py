# tests/unit/test_svl.py — Sonic Visualiser SVL 레이어 생성 테스트

import numpy as np
import pytest
from xml.etree.ElementTree import parse as parse_xml

from audioman.core.svl import (
    write_time_instants,
    write_time_values,
    write_notes,
    write_dense3d,
)


class TestWriteTimeInstants:
    def test_creates_valid_xml(self, tmp_path):
        path = tmp_path / "instants.svl"
        write_time_instants(path, frames=[0, 4410, 8820], sample_rate=44100)

        tree = parse_xml(str(path))
        root = tree.getroot()
        assert root.tag == "sv"

    def test_point_count(self, tmp_path):
        path = tmp_path / "instants.svl"
        frames = [0, 1000, 2000, 3000]
        write_time_instants(path, frames=frames)

        tree = parse_xml(str(path))
        points = tree.findall(".//point")
        assert len(points) == 4

    def test_point_frames(self, tmp_path):
        path = tmp_path / "instants.svl"
        frames = [100, 200, 300]
        write_time_instants(path, frames=frames)

        tree = parse_xml(str(path))
        points = tree.findall(".//point")
        frame_values = [int(p.get("frame")) for p in points]
        assert frame_values == [100, 200, 300]

    def test_labels(self, tmp_path):
        path = tmp_path / "instants.svl"
        frames = [0, 1000]
        labels = ["onset", "beat"]
        write_time_instants(path, frames=frames, labels=labels)

        tree = parse_xml(str(path))
        points = tree.findall(".//point")
        assert points[0].get("label") == "onset"
        assert points[1].get("label") == "beat"

    def test_empty_frames(self, tmp_path):
        path = tmp_path / "empty.svl"
        write_time_instants(path, frames=[])

        tree = parse_xml(str(path))
        points = tree.findall(".//point")
        assert len(points) == 0

    def test_sample_rate_in_model(self, tmp_path):
        path = tmp_path / "instants.svl"
        write_time_instants(path, frames=[0], sample_rate=48000)

        tree = parse_xml(str(path))
        model = tree.find(".//model")
        assert model.get("sampleRate") == "48000"


class TestWriteTimeValues:
    def test_creates_valid_xml(self, tmp_path):
        path = tmp_path / "values.svl"
        write_time_values(
            path, frames=[0, 512, 1024], values=[100.0, 200.0, 300.0],
            name="centroid", units="Hz",
        )
        tree = parse_xml(str(path))
        assert tree.getroot().tag == "sv"

    def test_point_values(self, tmp_path):
        path = tmp_path / "values.svl"
        frames = [0, 512]
        values = [440.0, 880.0]
        write_time_values(path, frames=frames, values=values)

        tree = parse_xml(str(path))
        points = tree.findall(".//point")
        assert float(points[0].get("value")) == pytest.approx(440.0)
        assert float(points[1].get("value")) == pytest.approx(880.0)

    def test_model_attributes(self, tmp_path):
        path = tmp_path / "values.svl"
        write_time_values(
            path, frames=[0], values=[1.5],
            name="energy", units="dB", sample_rate=96000,
        )
        tree = parse_xml(str(path))
        model = tree.find(".//model")
        assert model.get("name") == "energy"
        assert model.get("units") == "dB"
        assert model.get("sampleRate") == "96000"

    def test_min_max_in_model(self, tmp_path):
        path = tmp_path / "values.svl"
        values = [1.0, 5.0, 3.0]
        write_time_values(path, frames=[0, 512, 1024], values=values)

        tree = parse_xml(str(path))
        model = tree.find(".//model")
        assert float(model.get("minimum")) == pytest.approx(1.0)
        assert float(model.get("maximum")) == pytest.approx(5.0)


class TestWriteNotes:
    def test_creates_valid_xml(self, tmp_path):
        path = tmp_path / "notes.svl"
        write_notes(
            path,
            frames=[0, 22050],
            pitches=[60.0, 72.0],
            durations=[22050, 22050],
        )
        tree = parse_xml(str(path))
        assert tree.getroot().tag == "sv"

    def test_note_attributes(self, tmp_path):
        path = tmp_path / "notes.svl"
        write_notes(
            path,
            frames=[1000],
            pitches=[69.0],
            durations=[5000],
            levels=[0.8],
            labels=["A4"],
        )
        tree = parse_xml(str(path))
        point = tree.find(".//point")
        assert int(point.get("frame")) == 1000
        assert float(point.get("value")) == pytest.approx(69.0)
        assert int(point.get("duration")) == 5000
        assert float(point.get("level")) == pytest.approx(0.8)
        assert point.get("label") == "A4"

    def test_default_levels(self, tmp_path):
        path = tmp_path / "notes.svl"
        write_notes(path, frames=[0], pitches=[60.0], durations=[100])

        tree = parse_xml(str(path))
        point = tree.find(".//point")
        assert float(point.get("level")) == pytest.approx(1.0)

    def test_model_subtype_note(self, tmp_path):
        path = tmp_path / "notes.svl"
        write_notes(path, frames=[0], pitches=[60.0], durations=[100])

        tree = parse_xml(str(path))
        model = tree.find(".//model")
        assert model.get("subtype") == "note"


class TestWriteDense3d:
    def test_creates_valid_xml(self, tmp_path):
        path = tmp_path / "spec.svl"
        matrix = np.random.rand(10, 512).astype(np.float32)
        write_dense3d(path, matrix, sample_rate=44100, window_size=1024, hop_size=512)

        tree = parse_xml(str(path))
        assert tree.getroot().tag == "sv"

    def test_row_count(self, tmp_path):
        path = tmp_path / "spec.svl"
        n_frames, n_bins = 5, 64
        matrix = np.ones((n_frames, n_bins), dtype=np.float32)
        write_dense3d(path, matrix)

        tree = parse_xml(str(path))
        rows = tree.findall(".//row")
        assert len(rows) == n_frames

    def test_row_values(self, tmp_path):
        path = tmp_path / "spec.svl"
        matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        write_dense3d(path, matrix)

        tree = parse_xml(str(path))
        rows = tree.findall(".//row")
        # 첫 번째 row의 값 확인
        vals = [float(v) for v in rows[0].text.strip().split()]
        assert vals == pytest.approx([1.0, 2.0, 3.0], abs=1e-5)

    def test_bin_names(self, tmp_path):
        path = tmp_path / "spec.svl"
        matrix = np.ones((3, 2), dtype=np.float32)
        write_dense3d(path, matrix, bin_names=["0-100Hz", "100-200Hz"])

        tree = parse_xml(str(path))
        bins = tree.findall(".//bin")
        assert len(bins) == 2
        assert bins[0].get("name") == "0-100Hz"
        assert bins[1].get("name") == "100-200Hz"

    def test_model_min_max(self, tmp_path):
        path = tmp_path / "spec.svl"
        matrix = np.array([[0.1, 0.9], [0.3, 0.7]])
        write_dense3d(path, matrix)

        tree = parse_xml(str(path))
        model = tree.find(".//model")
        assert float(model.get("minimum")) == pytest.approx(0.1)
        assert float(model.get("maximum")) == pytest.approx(0.9)

    def test_model_ybin_count(self, tmp_path):
        path = tmp_path / "spec.svl"
        matrix = np.ones((4, 128), dtype=np.float32)
        write_dense3d(path, matrix)

        tree = parse_xml(str(path))
        model = tree.find(".//model")
        assert model.get("yBinCount") == "128"
