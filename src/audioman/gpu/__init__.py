"""GPU acceleration capability boundary.

GPU processing is not implemented yet.  Importing this module is safe, but
callers can explicitly check the capability instead of assuming acceleration
is available.
"""


class GPUAccelerationUnavailable(RuntimeError):
    """Raised when a GPU-only operation is requested before implementation."""


def is_available() -> bool:
    """Return whether Audioman currently provides GPU acceleration."""
    return False


def require_available() -> None:
    """Fail clearly for callers that require GPU acceleration."""
    raise GPUAccelerationUnavailable("GPU acceleration is not implemented in audioman")
