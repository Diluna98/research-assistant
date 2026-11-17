
import subprocess
import pytest
from gpu_telemetry import read_gpu_telemetry


def test_gpu_telemetry(monkeypatch):

    def fake_run(command, **kwargs):

        assert command[0] == "nvidia-smi"
        assert kwargs["timeout"] == 10

        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="NVIDIA GeForce RTX 4050 Laptop GPU, 0, 948, 6141, 41\n",
            stderr=""
        )

    monkeypatch.setattr(
        "gpu_telemetry.subprocess.run",
        fake_run
    )

    result = read_gpu_telemetry()

    assert result["name"] == "NVIDIA GeForce RTX 4050 Laptop GPU"
    assert result["utilization_percent"] == 0
    assert result["memory_used_mib"] == 948
    assert result["memory_total_mib"] == 6141
    assert result["temperature_c"] == 41


def test_gpu_tool_without_nvidia(monkeypatch):

    def fake_run(command, **kwargs):
        raise FileNotFoundError(
            "nvidia-smi"
        )

    monkeypatch.setattr(
        "gpu_telemetry.subprocess.run",
        fake_run
    )

    with pytest.raises(
        RuntimeError,
        match="nvidia-smi is not available"
    ):
        read_gpu_telemetry()



def test_gpu_command_failure(monkeypatch):

    def fake_run(command, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=command,
            stderr="NVIDIA driver communication failed"
        )

    monkeypatch.setattr(
        "gpu_telemetry.subprocess.run",
        fake_run
    )

    with pytest.raises(
        RuntimeError,
        match="nvidia-smi command failed"
    ):
        read_gpu_telemetry()


def test_gpu_command_timeout(monkeypatch):

    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=command,
            timeout=10
        )

    monkeypatch.setattr(
        "gpu_telemetry.subprocess.run",
        fake_run
    )

    with pytest.raises(
        RuntimeError,
        match="nvidia-smi timed out"
    ):
        read_gpu_telemetry()

