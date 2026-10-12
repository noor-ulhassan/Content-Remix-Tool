import subprocess
from types import SimpleNamespace

from src.ffmpeg import command_builder as command_builder_module
from src.ffmpeg.command_builder import CommandBuilder


def _fake_run(working_encoders):
    """Return a subprocess.run stand-in where only some encoders succeed."""

    def fake_run(cmd, **kwargs):
        encoder = cmd[cmd.index("-c:v") + 1]
        if encoder in working_encoders:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        # Generic failure text: no "unknown encoder" / "not found" in it
        return SimpleNamespace(
            returncode=1,
            stdout=b"",
            stderr=b"Error while opening encoder - maybe incorrect parameters",
        )

    return fake_run


def test_detect_gpu_encoders_excludes_encoders_that_fail(monkeypatch):
    monkeypatch.setattr(
        command_builder_module.subprocess,
        "run",
        _fake_run({"h264_amf", "hevc_amf"}),
    )

    builder = CommandBuilder()
    encoders = builder.detect_gpu_encoders()

    assert encoders == ["h264_amf", "hevc_amf"]
    assert builder._gpu_encoders == ["h264_amf", "hevc_amf"]


def test_detect_gpu_encoders_returns_empty_when_none_work(monkeypatch):
    monkeypatch.setattr(command_builder_module.subprocess, "run", _fake_run(set()))

    assert CommandBuilder().detect_gpu_encoders() == []


def test_detect_gpu_encoders_skips_encoders_that_time_out(monkeypatch):
    def fake_run(cmd, **kwargs):
        encoder = cmd[cmd.index("-c:v") + 1]
        if encoder == "h264_nvenc":
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(command_builder_module.subprocess, "run", fake_run)

    encoders = CommandBuilder().detect_gpu_encoders()

    assert "h264_nvenc" not in encoders
    assert "h264_amf" in encoders
