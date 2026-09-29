# Created: 2026-03-21
# Purpose: Batch processing utilities

from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".aiff", ".aif", ".ogg", ".opus", ".m4a", ".wma"}


def collect_audio_files(path: str | Path, recursive: bool = False) -> list[Path]:
    """Collect audio files from a directory"""
    path = Path(path)
    if path.is_file():
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Path not found: {path}")

    glob_pattern = "**/*" if recursive else "*"
    files = sorted(
        f for f in path.glob(glob_pattern)
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )
    return files


def resolve_output_path(
    input_path: Path,
    input_dir: Path,
    output_dir: Path,
    suffix: str = "",
    ext: str = ".wav",
) -> Path:
    """Build the output path matching an input file (preserving the directory structure)"""
    relative = input_path.relative_to(input_dir) if input_dir != input_path else input_path.name
    stem = Path(relative).stem
    parent = Path(relative).parent
    out_name = f"{stem}{suffix}{ext}"
    out_path = output_dir / parent / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path
