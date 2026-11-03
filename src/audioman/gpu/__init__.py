"""GPU acceleration capability boundary.

Audioman runs GPU work through torch (``torch.cuda`` / ``torch.backends.mps``);
see ``audioman.core.gpu_spectral``.  This module exposes that capability so
callers can check for a usable device instead of assuming one, and so that
GPU-only code paths can fail with a clear error on CPU-only machines.

Importing this module stays cheap: torch is imported lazily inside
``is_available``.
"""


class GPUAccelerationUnavailable(RuntimeError):
    """Raised when a GPU-only operation is requested but no GPU device is usable."""


def is_available() -> bool:
    """Return whether a GPU backend (CUDA or MPS) is usable via torch.

    Reports *device availability*, not "every audioman feature is accelerated":
    a ``True`` result means a caller may place tensors on a GPU device.
    """
    try:
        import torch
    except ImportError:
        return False

    if torch.cuda.is_available():
        return True

    try:
        return bool(torch.backends.mps.is_available())
    except (AttributeError, RuntimeError):
        return False


def require_available() -> None:
    """Fail clearly for callers that require GPU acceleration."""
    if not is_available():
        raise GPUAccelerationUnavailable(
            "GPU acceleration requires a usable CUDA or MPS device; "
            "none is available on this machine"
        )
