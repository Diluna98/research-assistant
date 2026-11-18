
from types import SimpleNamespace

from gpu_processes import read_gpu_processes


def test_gpu_process_discovery(monkeypatch):

    fake_output = (
        "1234, C:\\Program Files\\Ollama\\ollama.exe\n"
        "5678, [Insufficient Permissions]\n"
    )

    def fake_run(command, **kwargs):
        return SimpleNamespace(stdout=fake_output)

    monkeypatch.setattr(
        "gpu_processes.subprocess.run",
        fake_run
    )

    result = read_gpu_processes()

    assert result["count"] == 2

    assert result["processes"][0] == {
        "pid": 1234,
        "name": "ollama.exe"
    }

    assert result["processes"][1] == {
        "pid": 5678,
        "name": None
    }
