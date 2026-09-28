# Created: 2026-04-05
# Purpose: audioman bounce subcommand — multitrack bounce

import argparse

from audioman.cli.output import print_error, print_json, print_success, print_warning, output_console
from audioman.core.findings import json_envelope, schema_uri


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("bounce", help="Bounce multiple tracks into a single stereo file")
    parser.add_argument("inputs", nargs="*", help="Input audio files")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument(
        "--gain", default="",
        help="Comma-separated gain values in dB per track (e.g. '0,-3,-6')",
    )
    parser.add_argument(
        "--pan", default="",
        help="Comma-separated pan values per track (-1.0 L ~ 0.0 C ~ 1.0 R, e.g. '0,-0.5,0.5')",
    )
    parser.add_argument(
        "--chain", default="",
        help="Per-track plugin chains separated by '|' (e.g. 'denoise|limiter:threshold=-1|')",
    )
    parser.add_argument("--session", help="Session file (YAML/JSON) — overrides other track options")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.set_defaults(func=run)


def _parse_float_list(s: str) -> list[float]:
    """Parse a comma-separated list of floats."""
    if not s.strip():
        return []
    return [float(v.strip()) for v in s.split(",")]


def run(args: argparse.Namespace) -> None:
    from audioman.core.mixer import TrackConfig, bounce
    from audioman.core.pipeline import parse_chain_string

    # Session file mode
    if args.session:
        from audioman.core.session import load_session
        try:
            session = load_session(args.session)
        except Exception as e:
            print_error(f"Failed to load session file: {e}")
            return

        tracks = session.tracks
        output_path = args.output if args.output else session.output
        sample_rate = session.sample_rate
        subtype = session.subtype
    else:
        # CLI argument mode
        if not args.inputs:
            print_error("Specify input files (or use --session)")
            return

        try:
            gains = _parse_float_list(args.gain)
            pans = _parse_float_list(args.pan)
        except ValueError as e:
            print_error(f"--gain and --pan must be comma-separated numbers: {e}")
            return

        # Parse per-track chains (separated by '|')
        chains = []
        if args.chain.strip():
            for chain_str in args.chain.split("|"):
                chain_str = chain_str.strip()
                if chain_str:
                    chains.append(parse_chain_string(chain_str))
                else:
                    chains.append(None)

        tracks = []
        for i, inp in enumerate(args.inputs):
            tracks.append(TrackConfig(
                path=inp,
                gain_db=gains[i] if i < len(gains) else 0.0,
                pan=pans[i] if i < len(pans) else 0.0,
                chain=chains[i] if i < len(chains) else None,
            ))

        output_path = args.output
        sample_rate = None
        subtype = "PCM_24"

    # Dry-run
    if args.dry_run:
        plan = {
            "dry_run": True,
            "output": output_path,
            "track_count": len(tracks),
            "tracks": [t.to_dict() for t in tracks],
        }
        if args.json:
            print_json(json_envelope("bounce", plan, schema=schema_uri("bounce")))
        else:
            output_console.print(f"\n[bold]Bounce Plan[/bold] — {len(tracks)} tracks → {output_path}")
            for i, t in enumerate(tracks, 1):
                chain_str = f" → [{', '.join(s.plugin_name for s in t.chain)}]" if t.chain else ""
                output_console.print(
                    f"  {i}. {t.path}  gain={t.gain_db:+.1f}dB  pan={t.pan:+.1f}{chain_str}"
                )
        return

    # Execute
    try:
        result = bounce(
            tracks=tracks,
            output_path=output_path,
            sample_rate=sample_rate,
            subtype=subtype,
        )
    except Exception as e:
        print_error(f"Bounce failed: {e}")
        return

    if args.json:
        print_json(json_envelope("bounce", result.to_dict(), schema=schema_uri("bounce")))
        return

    print_success("Bounce complete")
    output_console.print(f"  Tracks: {result.track_count}")
    output_console.print(f"  Output: {result.output_path}")
    output_console.print(f"  SR:     {result.sample_rate} Hz")
    output_console.print(f"  Time:   {result.duration_seconds}s")
    if result.clipping_detected:
        print_warning("Clipping detected — lower the track levels or use a master limiter")
    print_success("Done")
