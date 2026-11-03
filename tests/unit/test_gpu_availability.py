# tests/unit/test_gpu_availability.py — audioman.gpu capability boundary tests.
#
# Deterministic: fakes torch in sys.modules, so results do not depend on the
# host's GPU hardware.

import importlib
import sys
from types import SimpleNamespace

import pytest

import audioman.gpu as gpu_mod
from audioman.gpu import GPUAccelerationUnavailable


def _fake_torch(cuda: bool, mps: bool):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


class TestIsAvailable:
    def test_cuda_available(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True, mps=False))
        assert gpu_mod.is_available() is True

    def test_mps_only(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False, mps=True))
        assert gpu_mod.is_available() is True

    def test_cpu_only(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False, mps=False))
        assert gpu_mod.is_available() is False

    def test_torch_missing(self, monkeypatch):
        """A poisoned sys.modules entry makes `import torch` raise ImportError."""
        monkeypatch.setitem(sys.modules, "torch", None)
        assert gpu_mod.is_available() is False

    def test_missing_mps_backend(self, monkeypatch):
        """torch builds without an mps attribute must not crash the probe."""
        fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        monkeypatch.setitem(sys.modules, "torch", fake)
        assert gpu_mod.is_available() is False


class TestRequireAvailable:
    def test_raises_when_unavailable(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False, mps=False))
        with pytest.raises(GPUAccelerationUnavailable):
            gpu_mod.require_available()

    def test_passes_when_available(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True, mps=False))
        assert gpu_mod.require_available() is None


class TestImportStaysCheap:
    def test_module_import_does_not_import_torch(self, monkeypatch):
        """Importing audioman.gpu must not pull in torch (lazy import inside the probe)."""
        monkeypatch.setitem(sys.modules, "torch", None)  # any import torch would fail now
        reloaded = importlib.reload(gpu_mod)
        assert reloaded.is_available() is False
