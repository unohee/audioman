# Created: 2026-03-23
# Purpose: 범용 VST 프리셋 파서 (FXP/FXB, vstpreset, aupreset)
# Dependencies: struct, zlib (표준 라이브러리만 사용)

import logging
import struct
import zlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# --- FXP/FXB 상수 ---
FXP_MAGIC = b"CcnK"
FXP_REGULAR_PRESET = b"FxCk"  # 파라미터 배열 방식
FXP_OPAQUE_PRESET = b"FPCh"  # opaque chunk 방식
FXB_REGULAR_BANK = b"FxBk"  # 뱅크 (파라미터 배열)
FXB_OPAQUE_BANK = b"FBCh"  # 뱅크 (opaque chunk)
FXP_HEADER_SIZE = 56  # 프리셋 이름 필드 시작 전까지

# --- vstpreset 상수 ---
VSTPRESET_MAGIC = b"VST3"


@dataclass
class PresetData:
    """파싱된 프리셋 데이터"""

    file_path: str
    format: str  # "fxp", "fxb", "vstpreset", "aupreset"
    plugin_id: str = ""
    preset_name: str = ""
    version: int = 0
    num_params: int = 0
    parameters: dict[str, Any] = field(default_factory=dict)
    chunk_data: Optional[bytes] = None  # opaque chunk (raw)
    chunk_size: int = 0
    is_opaque: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_chunk: bool = False) -> dict:
        d = asdict(self)
        if not include_chunk:
            d.pop("chunk_data", None)
        elif d.get("chunk_data"):
            # bytes를 hex 문자열로 변환
            d["chunk_data"] = d["chunk_data"].hex()
        return d


# =============================================================================
# FXP / FXB 파서
# =============================================================================


def parse_fxp(path: str | Path) -> PresetData:
    """VST2 FXP/FXB 프리셋 파일 파싱

    FXP 바이너리 구조 (Big-endian):
    - [0:4]   magic: 'CcnK'
    - [4:8]   byte_size: 전체 크기 - 8
    - [8:12]  fx_magic: 'FxCk' | 'FPCh' | 'FxBk' | 'FBCh'
    - [12:16] format_version: 1
    - [16:20] plugin_id: 고유 플러그인 ID (int32)
    - [20:24] plugin_version
    - [24:28] num_params (regular) 또는 num_programs (bank)
    - [28:56] preset_name: 28 바이트 null-terminated 문자열
    - [56:]   데이터 (파라미터 배열 또는 opaque chunk)
    """
    path = Path(path)
    data = path.read_bytes()

    if len(data) < 60:
        raise ValueError(f"FXP 파일이 너무 작습니다: {len(data)} bytes")

    magic = data[0:4]
    if magic != FXP_MAGIC:
        raise ValueError(f"FXP 매직 바이트 불일치: {magic!r} (expected {FXP_MAGIC!r})")

    byte_size = struct.unpack(">I", data[4:8])[0]
    fx_magic = data[8:12]
    format_version = struct.unpack(">I", data[12:16])[0]
    plugin_id = struct.unpack(">i", data[16:20])[0]
    plugin_version = struct.unpack(">I", data[20:24])[0]
    num_params = struct.unpack(">I", data[24:28])[0]

    # 프리셋 이름 (28 바이트, null-terminated)
    name_bytes = data[28:56]
    preset_name = name_bytes.split(b"\x00")[0].decode("ascii", errors="replace")

    result = PresetData(
        file_path=str(path),
        format="fxp" if fx_magic in (FXP_REGULAR_PRESET, FXP_OPAQUE_PRESET) else "fxb",
        plugin_id=f"0x{plugin_id & 0xFFFFFFFF:08X}",
        preset_name=preset_name,
        version=plugin_version,
        num_params=num_params,
        metadata={
            "format_version": format_version,
            "fx_magic": fx_magic.decode("ascii", errors="replace"),
            "byte_size": byte_size,
            "file_size": len(data),
        },
    )

    if fx_magic == FXP_REGULAR_PRESET:
        # 파라미터 배열: float32 × num_params (Big-endian)
        _parse_fxp_regular(data, 56, num_params, result)

    elif fx_magic == FXP_OPAQUE_PRESET:
        # opaque chunk
        _parse_fxp_opaque(data, 56, result)

    elif fx_magic == FXB_REGULAR_BANK:
        # 뱅크: 여러 프리셋 포함
        result.metadata["type"] = "bank"
        result.metadata["num_programs"] = num_params
        _parse_fxb_regular(data, 56, result)

    elif fx_magic == FXB_OPAQUE_BANK:
        # opaque 뱅크
        result.metadata["type"] = "bank_opaque"
        _parse_fxb_opaque(data, 56, result)

    else:
        raise ValueError(f"알 수 없는 FXP 타입: {fx_magic!r}")

    logger.debug(f"FXP 파싱 완료: {preset_name} ({num_params} params, opaque={result.is_opaque})")
    return result


def _parse_fxp_regular(data: bytes, offset: int, num_params: int, result: PresetData) -> None:
    """레귤러 프리셋: Big-endian float32 파라미터 배열"""
    result.is_opaque = False
    params = {}
    for i in range(num_params):
        pos = offset + i * 4
        if pos + 4 > len(data):
            break
        val = struct.unpack(">f", data[pos : pos + 4])[0]
        params[f"param_{i:03d}"] = round(val, 6)

    result.parameters = params
    result.num_params = len(params)


def _parse_fxp_opaque(data: bytes, offset: int, result: PresetData) -> None:
    """Opaque chunk 프리셋: chunk_size + raw binary"""
    result.is_opaque = True

    if offset + 4 > len(data):
        return

    chunk_size = struct.unpack(">I", data[offset : offset + 4])[0]
    chunk_data = data[offset + 4 : offset + 4 + chunk_size]
    result.chunk_size = chunk_size
    result.chunk_data = chunk_data

    # zlib 압축 해제 시도 (Serum 등 일부 플러그인)
    params = _try_extract_chunk_params(chunk_data, result)
    if params:
        result.parameters = params
        result.num_params = len(params)


def _parse_fxb_regular(data: bytes, offset: int, result: PresetData) -> None:
    """레귤러 뱅크: 여러 프리셋을 포함"""
    result.is_opaque = False
    # 뱅크는 프리셋 목록으로 구성 — 메타데이터만 기록
    result.metadata["note"] = "bank contains multiple presets (use --expand for details)"


def _parse_fxb_opaque(data: bytes, offset: int, result: PresetData) -> None:
    """Opaque 뱅크"""
    result.is_opaque = True
    if offset + 4 > len(data):
        return
    chunk_size = struct.unpack(">I", data[offset : offset + 4])[0]
    chunk_data = data[offset + 4 : offset + 4 + chunk_size]
    result.chunk_size = chunk_size
    result.chunk_data = chunk_data


def _try_extract_chunk_params(chunk: bytes, result: PresetData) -> dict[str, float] | None:
    """Opaque chunk에서 파라미터 추출 시도

    1) zlib 압축 해제 시도
    2) 알려진 플러그인 오프셋 시도 (Serum 등)
    3) 휴리스틱 float32 배열 탐색
    """
    decompressed = None

    # zlib 압축 해제 시도
    try:
        decompressed = zlib.decompress(chunk)
        result.metadata["compression"] = "zlib"
        result.metadata["decompressed_size"] = len(decompressed)
    except zlib.error:
        # 압축되지 않은 chunk — 원본 데이터 사용
        decompressed = chunk

    if decompressed is None or len(decompressed) < 4:
        return None

    # 알려진 플러그인별 파라미터 오프셋 시도
    params = _try_known_plugin_offsets(decompressed, result)
    if params:
        return params

    # 범용 float32 배열 휴리스틱 탐색
    params = _scan_float_params(decompressed)
    if params and len(params) >= 4:
        return params

    return None


# 알려진 플러그인 파라미터 오프셋 테이블
# (plugin_id, param_offset, endian, 설명)
KNOWN_PLUGIN_OFFSETS = {
    "0x58667358": {  # Xfer Serum (v1)
        "name": "Serum",
        "param_offset": 0x3460,
        "endian": "<f",  # little-endian
    },
}


def _try_known_plugin_offsets(data: bytes, result: PresetData) -> dict[str, float] | None:
    """알려진 플러그인의 파라미터 오프셋으로 추출 시도"""
    plugin_id = result.plugin_id
    info = KNOWN_PLUGIN_OFFSETS.get(plugin_id)

    if not info:
        return None

    offset = info["param_offset"]
    endian = info["endian"]

    if offset >= len(data):
        return None

    result.metadata["plugin_detected"] = info["name"]
    result.metadata["param_offset"] = f"0x{offset:04X}"

    # 해당 오프셋에서 연속된 0-1 범위 float 추출
    params = {}
    i = 0
    while offset + i * 4 + 4 <= len(data):
        pos = offset + i * 4
        val = struct.unpack(endian, data[pos : pos + 4])[0]
        if -0.01 <= val <= 1.01 and not _is_nan_or_inf(val):
            params[f"param_{i:03d}"] = round(val, 6)
            i += 1
        else:
            break

    if len(params) >= 10:  # 최소 10개 이상이면 유효
        logger.debug(f"{info['name']} 파라미터 감지: offset=0x{offset:04X}, count={len(params)}")
        return params

    return None


def _scan_float_params(data: bytes) -> dict[str, float] | None:
    """바이너리 데이터에서 연속된 정규화 float32 파라미터 영역 탐색

    전략: 4바이트 정렬된 위치에서 little-endian float32를 읽어
    0.0~1.0 범위 값이 연속으로 나타나는 최장 구간 선택.
    """
    if len(data) < 16:
        return None

    best_start = 0
    best_count = 0

    # little-endian 먼저 시도
    for endian, fmt in [("<f", "le"), (">f", "be")]:
        i = 0
        while i <= len(data) - 4:
            count = 0
            start = i
            while i <= len(data) - 4:
                val = struct.unpack(endian, data[i : i + 4])[0]
                # 정규화된 파라미터: -0.01 ~ 1.01 허용 (부동소수점 오차)
                if -0.01 <= val <= 1.01 and not _is_nan_or_inf(val):
                    count += 1
                    i += 4
                else:
                    break

            if count > best_count:
                best_count = count
                best_start = start
                best_endian = endian

            i = start + 4  # 다음 오프셋

    if best_count < 4:
        return None

    params = {}
    for j in range(best_count):
        pos = best_start + j * 4
        val = struct.unpack(best_endian, data[pos : pos + 4])[0]
        params[f"param_{j:03d}"] = round(val, 6)

    return params


def _is_nan_or_inf(val: float) -> bool:
    """NaN 또는 Inf 체크"""
    import math

    return math.isnan(val) or math.isinf(val)


# =============================================================================
# VST3 Preset 파서
# =============================================================================


def parse_vstpreset(path: str | Path) -> PresetData:
    """.vstpreset 파일 파싱

    VST3 프리셋 구조:
    - [0:4]   magic: 'VST3'
    - [4:8]   version
    - [8:40]  class_id (32 bytes ASCII hex)
    - [40:48] list_offset (데이터 끝 → 속성 리스트 시작)
    - [48:]   chunk data ... 속성 리스트
    """
    path = Path(path)
    data = path.read_bytes()

    if len(data) < 48:
        raise ValueError(f"vstpreset 파일이 너무 작습니다: {len(data)} bytes")

    magic = data[0:4]
    if magic != VSTPRESET_MAGIC:
        raise ValueError(f"vstpreset 매직 바이트 불일치: {magic!r}")

    version = struct.unpack("<I", data[4:8])[0]
    class_id = data[8:40].decode("ascii", errors="replace").strip("\x00")
    list_offset = struct.unpack("<q", data[40:48])[0]

    result = PresetData(
        file_path=str(path),
        format="vstpreset",
        plugin_id=class_id,
        version=version,
        metadata={
            "file_size": len(data),
            "list_offset": list_offset,
        },
    )

    # chunk 데이터 추출 (헤더 이후 ~ list_offset)
    chunk_start = 48
    chunk_end = min(list_offset, len(data)) if list_offset > 48 else len(data)
    chunk_data = data[chunk_start:chunk_end]
    result.chunk_data = chunk_data
    result.chunk_size = len(chunk_data)
    result.is_opaque = True

    # 속성 리스트 파싱 (list_offset 이후)
    if list_offset > 0 and list_offset < len(data):
        attrs = _parse_vstpreset_attributes(data, list_offset)
        if attrs:
            result.metadata["attributes"] = attrs
            if "Preset Name" in attrs:
                result.preset_name = attrs["Preset Name"]

    # chunk에서 파라미터 추출 시도
    params = _try_extract_chunk_params(chunk_data, result)
    if params:
        result.parameters = params
        result.num_params = len(params)

    logger.debug(f"vstpreset 파싱 완료: {class_id} ({result.chunk_size} bytes chunk)")
    return result


def _parse_vstpreset_attributes(data: bytes, offset: int) -> dict[str, str]:
    """vstpreset 속성 리스트 파싱

    속성 리스트 구조:
    - [0:4] 'List'
    - [4:8] entry_count (uint32 LE)
    - 각 entry: id (128 bytes) + offset (uint64 LE) + size (uint64 LE)
    """
    attrs = {}

    if offset + 8 > len(data):
        return attrs

    list_magic = data[offset : offset + 4]
    if list_magic != b"List":
        return attrs

    entry_count = struct.unpack("<I", data[offset + 4 : offset + 8])[0]

    pos = offset + 8
    entries = []
    for _ in range(entry_count):
        if pos + 144 > len(data):
            break
        attr_id = data[pos : pos + 128].split(b"\x00")[0].decode("utf-8", errors="replace")
        attr_offset = struct.unpack("<Q", data[pos + 128 : pos + 136])[0]
        attr_size = struct.unpack("<Q", data[pos + 136 : pos + 144])[0]
        entries.append((attr_id, attr_offset, attr_size))
        pos += 144

    # 속성 값 읽기
    for attr_id, attr_offset, attr_size in entries:
        if attr_offset + attr_size <= len(data):
            val_bytes = data[attr_offset : attr_offset + attr_size]
            try:
                # UTF-16 LE 디코딩 시도 (vstpreset 문자열 표준)
                val = val_bytes.decode("utf-16-le").strip("\x00")
            except (UnicodeDecodeError, ValueError):
                try:
                    val = val_bytes.decode("utf-8", errors="replace").strip("\x00")
                except Exception:
                    val = val_bytes.hex()
            attrs[attr_id] = val

    return attrs


# =============================================================================
# AU Preset 파서 (macOS)
# =============================================================================


def parse_aupreset(path: str | Path) -> PresetData:
    """.aupreset 파일 파싱 (Apple plist 기반)"""
    path = Path(path)

    try:
        import plistlib
    except ImportError:
        raise ImportError("plistlib 필요 (Python 표준 라이브러리, macOS)")

    with open(path, "rb") as f:
        plist = plistlib.load(f)

    result = PresetData(
        file_path=str(path),
        format="aupreset",
        metadata={"plist_keys": list(plist.keys())},
    )

    # 표준 AU 프리셋 키
    if "name" in plist:
        result.preset_name = str(plist["name"])
    if "manufacturer" in plist:
        result.metadata["manufacturer"] = plist["manufacturer"]
    if "type" in plist:
        result.metadata["au_type"] = plist["type"]
    if "subtype" in plist:
        result.plugin_id = str(plist["subtype"])
    if "version" in plist:
        result.version = plist["version"]

    # 파라미터 데이터 추출
    if "data" in plist:
        # AU state data (바이너리)
        state_data = plist["data"]
        if isinstance(state_data, bytes):
            result.chunk_data = state_data
            result.chunk_size = len(state_data)
            result.is_opaque = True

            params = _try_extract_chunk_params(state_data, result)
            if params:
                result.parameters = params
                result.num_params = len(params)

    # 일부 AU 프리셋은 파라미터를 딕셔너리로 직접 저장
    if "ParameterValues" in plist:
        pv = plist["ParameterValues"]
        if isinstance(pv, dict):
            result.parameters = {str(k): v for k, v in pv.items()}
            result.num_params = len(result.parameters)
            result.is_opaque = False

    logger.debug(f"aupreset 파싱 완료: {result.preset_name}")
    return result


# =============================================================================
# Vital 프리셋 파서 (.vital — JSON)
# =============================================================================


def parse_vital(path: str | Path) -> PresetData:
    """.vital 프리셋 파일 파싱 (순수 JSON)

    구조: {"preset_name", "author", "settings": {파라미터 772개+}, "synth_version", ...}
    """
    path = Path(path)

    import json as _json
    data = _json.loads(path.read_text(encoding="utf-8"))

    result = PresetData(
        file_path=str(path),
        format="vital",
        preset_name=data.get("preset_name", path.stem),
        plugin_id="Vital",
        is_opaque=False,
    )

    # 메타데이터
    if "author" in data:
        result.metadata["author"] = data["author"]
    if "preset_style" in data:
        result.metadata["style"] = data["preset_style"]
    if "synth_version" in data:
        result.version = data["synth_version"]
        result.metadata["synth_version"] = data["synth_version"]
    if "comments" in data:
        result.metadata["comments"] = data["comments"]
    for m in ("macro1", "macro2", "macro3", "macro4"):
        if m in data and data[m]:
            result.metadata[m] = data[m]

    # settings에서 numeric 파라미터만 추출
    settings = data.get("settings", {})
    parameters = {}
    for key, value in settings.items():
        if isinstance(value, (int, float)):
            parameters[key] = round(float(value), 6)
        # 문자열/리스트 등은 메타데이터로

    result.parameters = parameters
    result.num_params = len(parameters)

    logger.debug(f"vital 파싱 완료: {result.preset_name} ({len(parameters)} params)")
    return result


# =============================================================================
# Phase Plant 프리셋 파서 (.phaseplant — 바이너리 + 내장 JSON 메타)
# =============================================================================


def parse_phaseplant(path: str | Path) -> PresetData:
    """.phaseplant 프리셋 파일 파싱

    바이너리 포맷이지만 내부에 JSON 메타데이터 블록이 포함.
    파라미터 추출은 DawDreamer 필요 (바이너리 구조가 독자적).
    """
    path = Path(path)
    data = path.read_bytes()

    result = PresetData(
        file_path=str(path),
        format="phaseplant",
        preset_name=path.stem,
        plugin_id="PhasePlant",
        is_opaque=True,
        chunk_size=len(data),
    )

    # 내장 JSON 메타 추출
    json_start = data.find(b"{")
    if json_start >= 0:
        # JSON 끝 찾기 (첫 번째 중괄호 쌍)
        depth = 0
        json_end = json_start
        for i in range(json_start, len(data)):
            if data[i:i+1] == b"{":
                depth += 1
            elif data[i:i+1] == b"}":
                depth -= 1
                if depth == 0:
                    json_end = i + 1
                    break

        try:
            import json as _json
            meta = _json.loads(data[json_start:json_end])
            if "description" in meta:
                result.metadata["description"] = meta["description"]
            if "author" in meta:
                result.metadata["author"] = meta["author"]
        except Exception:
            pass

    logger.debug(f"phaseplant 파싱 완료: {result.preset_name}")
    return result


# =============================================================================
# u-he H2P 프리셋 파서 (Diva, Bazille, Hive, Repro 등)
# =============================================================================


def parse_h2p(path: str | Path) -> PresetData:
    """u-he .h2p 프리셋 파일 파싱

    텍스트 기반 포맷:
    - /*@Meta ... */ — 메타데이터 블록
    - #AM=PluginName — 플러그인 이름
    - #cm=SectionName — 섹션 구분
    - Key=Value — 파라미터 (float 또는 int)
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    result = PresetData(
        file_path=str(path),
        format="h2p",
        preset_name=path.stem,
    )

    # 메타데이터 블록 파싱 /*@Meta ... */
    meta_block = {}
    import re
    meta_match = re.search(r"/\*@Meta\s*(.*?)\*/", text, re.DOTALL)
    if meta_match:
        meta_text = meta_match.group(1)
        current_key = None
        for line in meta_text.strip().splitlines():
            line = line.strip()
            if line.endswith(":"):
                current_key = line[:-1].strip()
            elif current_key and line.startswith("'") and line.endswith("'"):
                meta_block[current_key] = line.strip("'")
                current_key = None

    if meta_block:
        result.metadata["meta"] = meta_block
        if "Author" in meta_block:
            result.metadata["author"] = meta_block["Author"]

    # 본문 파싱
    parameters = {}
    current_section = ""
    plugin_name = ""

    for line in text.splitlines():
        line = line.strip()

        # 빈 줄, 주석 건너뛰기
        if not line or line.startswith("//") or line.startswith("/*"):
            continue

        # 지시어
        if line.startswith("#"):
            if line.startswith("#AM="):
                plugin_name = line[4:]
                result.plugin_id = plugin_name
            elif line.startswith("#cm="):
                current_section = line[4:]
            elif line.startswith("#Vers="):
                try:
                    result.version = int(line[6:])
                except ValueError:
                    pass
            continue

        # 파라미터 (Key=Value)
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()

            # 섹션 접두사 추가
            full_key = f"{current_section}.{key}" if current_section else key

            # 값 파싱
            try:
                # float 시도
                fval = float(value)
                parameters[full_key] = fval
            except ValueError:
                # 문자열 값
                parameters[full_key] = value

    result.parameters = parameters
    result.num_params = len(parameters)
    result.is_opaque = False

    if plugin_name:
        result.metadata["plugin_name"] = plugin_name

    logger.debug(f"h2p 파싱 완료: {result.preset_name} ({len(parameters)} params, plugin={plugin_name})")
    return result


# =============================================================================
# 자동 감지 + 배치 파싱
# =============================================================================

SUPPORTED_EXTENSIONS = {".fxp", ".fxb", ".vstpreset", ".aupreset", ".h2p", ".vital", ".phaseplant"}


def parse_auto(path: str | Path) -> PresetData:
    """확장자 기반 자동 파서 선택"""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"프리셋 파일 없음: {path}")

    ext = path.suffix.lower()

    if ext in (".fxp", ".fxb"):
        return parse_fxp(path)
    elif ext == ".vstpreset":
        return parse_vstpreset(path)
    elif ext == ".aupreset":
        return parse_aupreset(path)
    elif ext == ".h2p":
        return parse_h2p(path)
    elif ext == ".vital":
        return parse_vital(path)
    elif ext == ".phaseplant":
        return parse_phaseplant(path)
    else:
        raise ValueError(f"지원하지 않는 프리셋 포맷: {ext} (지원: {', '.join(sorted(SUPPORTED_EXTENSIONS))})")


def find_presets(directory: str | Path, recursive: bool = True) -> list[Path]:
    """디렉토리에서 프리셋 파일 검색"""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"디렉토리가 아닙니다: {directory}")

    presets = []
    pattern = "**/*" if recursive else "*"
    for p in directory.glob(pattern):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            presets.append(p)

    return sorted(presets)


# =============================================================================
# DawDreamer 기반 파라미터 이름 매핑
# =============================================================================

# 플러그인 ID → VST3 경로 매핑 (macOS 기본 경로)
KNOWN_PLUGIN_PATHS = {
    "0x58667358": "/Library/Audio/Plug-Ins/VST3/Serum.vst3",  # Xfer Serum v1
    "Diva": "/Library/Audio/Plug-Ins/VST3/Diva.vst3",  # u-he Diva
    "Bazille": "/Library/Audio/Plug-Ins/VST3/Bazille.vst3",  # u-he Bazille
    "Vital": "/Library/Audio/Plug-Ins/VST3/Vital.vst3",  # Vital
}

# 이름 매핑 캐시 (플러그인 경로 → 이름 리스트)
_param_name_cache: dict[str, list[str]] = {}


def get_param_names(plugin_path: str) -> list[str] | None:
    """DawDreamer로 플러그인의 파라미터 이름 목록 추출 (캐시됨)"""
    if plugin_path in _param_name_cache:
        return _param_name_cache[plugin_path]

    try:
        import dawdreamer as daw
    except ImportError:
        return None

    try:
        engine = daw.RenderEngine(44100, 512)
        synth = engine.make_plugin_processor("resolver", plugin_path)
        count = synth.get_plugin_parameter_size()
        names = [synth.get_parameter_name(i) for i in range(count)]
        _param_name_cache[plugin_path] = names
        return names
    except Exception as e:
        logger.warning(f"파라미터 이름 추출 실패: {e}")
        return None


def resolve_param_names(preset: PresetData, plugin_path: str | None = None) -> PresetData:
    """프리셋의 param_NNN 키를 실제 파라미터 이름으로 변환

    plugin_path가 None이면 plugin_id로 자동 감지 시도.
    """
    if not preset.parameters:
        return preset

    # 이미 이름이 있으면 (param_NNN이 아니면) 건너뛰기
    first_key = next(iter(preset.parameters))
    if not first_key.startswith("param_"):
        return preset

    # 플러그인 경로 결정
    if plugin_path is None:
        plugin_path = KNOWN_PLUGIN_PATHS.get(preset.plugin_id)

    if plugin_path is None:
        return preset

    from pathlib import Path
    if not Path(plugin_path).exists():
        return preset

    names = get_param_names(plugin_path)
    if names is None:
        return preset

    # param_NNN → 실제 이름으로 매핑
    named_params = {}
    for key, value in preset.parameters.items():
        if key.startswith("param_"):
            try:
                idx = int(key.split("_")[1])
                if idx < len(names):
                    named_params[names[idx]] = value
                else:
                    named_params[key] = value
            except (ValueError, IndexError):
                named_params[key] = value
        else:
            named_params[key] = value

    preset.parameters = named_params
    preset.metadata["param_names_resolved"] = True
    preset.metadata["param_name_source"] = plugin_path
    return preset
