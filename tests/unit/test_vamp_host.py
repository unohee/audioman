# tests/unit/test_vamp_host.py — core/vamp_host.py.
#
# The optional `vamp` package (pyproject extra `vamp`) and the Vamp plugin SDK are
# not installed on this host, so the plugin-dependent entry points are exercised by
# injecting a stub `vamp` module into sys.modules. The pure result-parsing helpers
# need no vamp at all and are tested with synthetic VampResult objects.
#
# One integration test is skipif-guarded on the real `vamp` package being present.

from __future__ import annotations

import sys

import numpy as np
import pytest

from audioman.core import vamp_host


class _StubVamp:
    """Minimal stand-in for the `vamp` bindings."""

    def __init__(self, collect_result=None, plugins=None, outputs=None):
        self._collect_result = collect_result
        self._plugins = plugins if plugins is not None else ["lib:plugB", "lib:plugA"]
        self._outputs = outputs if outputs is not None else {"o1": {"identifier": "o1"}}
        self.collect_calls: list = []

    def list_plugins(self):
        return list(self._plugins)

    def get_outputs_of(self, plugin_id):
        return self._outputs

    def collect(self, data, sample_rate, plugin_id, **kwargs):
        self.collect_calls.append((data, sample_rate, plugin_id, kwargs))
        return self._collect_result


@pytest.fixture
def stub_vamp(monkeypatch):
    def _install(stub):
        monkeypatch.setitem(sys.modules, "vamp", stub)
        return stub

    return _install


class TestImportVamp:
    def test_missing_package_raises_install_hint(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "vamp", None)
        # `import vamp` with a None entry raises ImportError → the helper's own message
        with pytest.raises(ImportError, match="vamp 패키지가 필요합니다"):
            vamp_host._import_vamp()

    def test_returns_injected_module(self, stub_vamp):
        stub = stub_vamp(_StubVamp())
        assert vamp_host._import_vamp() is stub


class TestListPlugins:
    def test_sorted(self, stub_vamp):
        stub_vamp(_StubVamp(plugins=["lib:z", "lib:a", "lib:m"]))
        assert vamp_host.list_plugins() == ["lib:a", "lib:m", "lib:z"]


class TestGetPluginOutputs:
    def test_returns_outputs_dict(self, stub_vamp):
        stub = stub_vamp(_StubVamp(outputs={"pitch": {"identifier": "pitch"}}))
        assert vamp_host.get_plugin_outputs("lib:plug") == {"pitch": {"identifier": "pitch"}}
        assert stub._outputs == {"pitch": {"identifier": "pitch"}}


class TestRunPlugin:
    def test_vector_result_and_default_kwargs(self, stub_vamp):
        stub = stub_vamp(_StubVamp(collect_result={"vector": (0.5, [1.0, 2.0])}))
        audio = np.ones((2, 1000), dtype=np.float32)
        result = vamp_host.run_plugin(audio, 48000, "lib:plug")
        assert result.shape == "vector"
        assert result.plugin_id == "lib:plug"
        assert result.output == ""
        assert result.sample_rate == 48000
        # mono conversion happened and no optional kwargs were passed
        data, sr, pid, kwargs = stub.collect_calls[0]
        assert data.ndim == 1
        assert kwargs == {}

    def test_mono_input_not_converted(self, stub_vamp):
        stub = stub_vamp(_StubVamp(collect_result={"list": []}))
        audio = np.ones(500, dtype=np.float32)
        result = vamp_host.run_plugin(audio, 44100, "lib:plug")
        assert result.shape == "list"
        assert stub.collect_calls[0][0].ndim == 1

    def test_three_part_plugin_id_splits_output(self, stub_vamp):
        stub = stub_vamp(_StubVamp(collect_result={"matrix": (0.1, np.zeros((4, 3, 2)))}))
        result = vamp_host.run_plugin(np.ones(200, dtype=np.float32), 48000, "lib:plug:pitch")
        assert result.plugin_id == "lib:plug"
        assert result.output == "pitch"
        assert result.shape == "matrix"
        assert stub.collect_calls[0][3]["output"] == "pitch"

    def test_explicit_output_wins_over_id_suffix(self, stub_vamp):
        stub = stub_vamp(_StubVamp(collect_result={"vector": (0.1, [0.0])}))
        result = vamp_host.run_plugin(
            np.ones(200, dtype=np.float32), 48000, "lib:plug:pitch", output="energy"
        )
        assert result.plugin_id == "lib:plug:pitch"  # kept intact when output given
        assert result.output == "energy"

    def test_all_optional_kwargs_forwarded(self, stub_vamp):
        stub = stub_vamp(_StubVamp(collect_result={"vector": (0.1, [0.0])}))
        vamp_host.run_plugin(
            np.ones(200, dtype=np.float32), 48000, "lib:plug",
            output="o", parameters={"p": 1.0}, block_size=2048, step_size=512,
        )
        kwargs = stub.collect_calls[0][3]
        assert kwargs == {"output": "o", "parameters": {"p": 1.0},
                          "block_size": 2048, "step_size": 512}

    def test_zero_block_and_step_omitted(self, stub_vamp):
        stub = stub_vamp(_StubVamp(collect_result={"vector": (0.1, [0.0])}))
        vamp_host.run_plugin(
            np.ones(200, dtype=np.float32), 48000, "lib:plug", block_size=0, step_size=0
        )
        assert stub.collect_calls[0][3] == {}

    def test_unknown_shape_reported(self, stub_vamp):
        stub_vamp(_StubVamp(collect_result={"something_else": 1}))
        result = vamp_host.run_plugin(np.ones(200, dtype=np.float32), 48000, "lib:plug")
        assert result.shape == "unknown"


class TestResultToFramesAndValues:
    def _res(self, shape, data, sr=48000):
        return vamp_host.VampResult("lib:plug", "", shape, sr, data)

    def test_vector_uses_step_as_hop(self):
        res = self._res("vector", {"vector": (0.5, [1.0, 2.0, 3.0])}, sr=48000)
        frames, values = vamp_host.result_to_frames_and_values(res, 48000)
        assert frames == [0, 24000, 48000]
        assert values == [1.0, 2.0, 3.0]

    def test_vector_zero_step_falls_back_to_hop_size(self):
        res = self._res("vector", {"vector": (0.0, [1.0, 2.0])}, sr=48000)
        frames, values = vamp_host.result_to_frames_and_values(res, 48000, hop_size=512)
        assert frames == [0, 512]

    def test_list_uses_timestamp_and_first_value(self):
        res = self._res("list", {"list": [
            {"timestamp": 0.0, "values": [5.0]},
            {"time": 1.0, "values": [7.0]},
        ]}, sr=48000)
        frames, values = vamp_host.result_to_frames_and_values(res, 48000)
        assert frames == [0, 48000]
        assert values == [5.0, 7.0]

    def test_list_missing_values_defaults_zero(self):
        res = self._res("list", {"list": [{"timestamp": 0.5}]}, sr=1000)
        frames, values = vamp_host.result_to_frames_and_values(res, 1000)
        assert frames == [500]
        assert values == [0.0]

    def test_matrix_shape_rejected(self):
        res = self._res("matrix", {"matrix": (0.1, np.zeros((2, 2)))})
        with pytest.raises(ValueError, match="vector/list 변환 불가"):
            vamp_host.result_to_frames_and_values(res, 48000)

    def test_unknown_shape_rejected(self):
        res = self._res("unknown", {})
        with pytest.raises(ValueError, match="shape=unknown"):
            vamp_host.result_to_frames_and_values(res, 48000)


class TestResultToInstants:
    def test_frames_and_labels(self):
        res = vamp_host.VampResult("p", "", "list", 1000, {"list": [
            {"timestamp": 0.25, "label": "A"},
            {"time": 0.75, "label": "B"},
        ]})
        frames, labels = vamp_host.result_to_instants(res, 1000)
        assert frames == [250, 750]
        assert labels == ["A", "B"]

    def test_missing_label_becomes_empty_string(self):
        res = vamp_host.VampResult("p", "", "list", 1000, {"list": [{"timestamp": 0.0}]})
        frames, labels = vamp_host.result_to_instants(res, 1000)
        assert labels == [""]

    def test_non_list_rejected(self):
        res = vamp_host.VampResult("p", "", "vector", 1000, {"vector": (0.1, [1.0])})
        with pytest.raises(ValueError, match="list 결과만 지원"):
            vamp_host.result_to_instants(res, 1000)


class TestResultToMatrix:
    def test_matrix_and_hop_samples(self):
        matrix = np.arange(24, dtype=np.float64).reshape(4, 3, 2)
        res = vamp_host.VampResult("p", "o", "matrix", 48000, {"matrix": (0.25, matrix)})
        out, hop = vamp_host.result_to_matrix(res)
        assert out.shape == (4, 3, 2)
        assert hop == 12000

    def test_non_matrix_rejected(self):
        res = vamp_host.VampResult("p", "", "vector", 48000, {"vector": (0.1, [1.0])})
        with pytest.raises(ValueError, match="matrix 변환 불가"):
            vamp_host.result_to_matrix(res)


def _vamp_installed() -> bool:
    try:
        import vamp  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not _vamp_installed(),
    reason="optional `vamp` package not installed (pyproject extra: vamp)",
)
class TestRealVampIntegration:
    def test_import_and_list_have_no_plugins_requirement(self):
        # list_plugins must return a list even when no Vamp plugins are installed
        assert isinstance(vamp_host.list_plugins(), list)
