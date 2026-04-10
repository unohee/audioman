# tests/unit/test_pipeline.py — 파이프라인 파싱 + 데이터 구조 테스트

import pytest

from audioman.core.pipeline import (
    PipelineStep,
    PipelineResult,
    parse_chain_string,
)


class TestPipelineStep:
    def test_to_dict(self):
        step = PipelineStep(plugin_name="denoise", params={"threshold": -20.0})
        d = step.to_dict()
        assert d == {"plugin": "denoise", "params": {"threshold": -20.0}}

    def test_to_dict_empty_params(self):
        step = PipelineStep(plugin_name="declick", params={})
        d = step.to_dict()
        assert d == {"plugin": "declick", "params": {}}


class TestParseChainString:
    """체인 문자열 파싱"""

    def test_single_plugin(self):
        steps = parse_chain_string("denoise")
        assert len(steps) == 1
        assert steps[0].plugin_name == "denoise"
        assert steps[0].params == {}

    def test_single_with_params(self):
        steps = parse_chain_string("denoise:threshold=-20")
        assert len(steps) == 1
        assert steps[0].plugin_name == "denoise"
        assert steps[0].params == {"threshold": -20.0}

    def test_multiple_plugins(self):
        steps = parse_chain_string("denoise,declick,dehum")
        assert len(steps) == 3
        assert [s.plugin_name for s in steps] == ["denoise", "declick", "dehum"]

    def test_multiple_with_params(self):
        steps = parse_chain_string("denoise:threshold=-20,dehum:freq=60")
        assert len(steps) == 2
        assert steps[0].params == {"threshold": -20.0}
        assert steps[1].params == {"freq": 60.0}

    def test_semicolon_separated_params(self):
        steps = parse_chain_string("eq:low=100;high=8000;gain=-3")
        assert len(steps) == 1
        assert steps[0].params == {"low": 100.0, "high": 8000.0, "gain": -3.0}

    def test_boolean_params(self):
        steps = parse_chain_string("limiter:enabled=true;lookahead=false")
        assert steps[0].params["enabled"] is True
        assert steps[0].params["lookahead"] is False

    def test_string_params(self):
        steps = parse_chain_string("preset:name=vocal_clean")
        assert steps[0].params["name"] == "vocal_clean"

    def test_empty_string(self):
        steps = parse_chain_string("")
        assert steps == []

    def test_whitespace_handling(self):
        steps = parse_chain_string("  denoise , declick  ")
        assert len(steps) == 2
        assert steps[0].plugin_name == "denoise"
        assert steps[1].plugin_name == "declick"

    def test_trailing_comma(self):
        steps = parse_chain_string("denoise,")
        assert len(steps) == 1


class TestPipelineResult:
    def test_to_dict(self):
        result = PipelineResult(
            input_path="/in.wav",
            output_path="/out.wav",
            steps=[{"plugin": "denoise", "params": {}}],
            input_stats={"peak": 0.5},
            output_stats={"peak": 0.4},
            duration_seconds=1.23,
        )
        d = result.to_dict()
        assert d["input_path"] == "/in.wav"
        assert d["duration_seconds"] == 1.23
        assert isinstance(d["steps"], list)
