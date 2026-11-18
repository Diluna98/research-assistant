
import json
import pytest

from gpu_tool_dispatcher import execute_tool


def test_executes_approved_tool(monkeypatch):

    def fake_telemetry():
        return {
            "utilization_percent": 75
        }

    monkeypatch.setattr(
        "gpu_tool_dispatcher.read_gpu_telemetry",
        fake_telemetry
    )

    result = execute_tool({
        "name": "read_gpu_telemetry",
        "arguments": {}
    })

    assert json.loads(result) == {
        "utilization_percent": 75
    }


def test_rejects_unknown_tool():

    with pytest.raises(
        ValueError,
        match="Unsupported tool request"
    ):
        execute_tool({
            "name": "delete_files",
            "arguments": {}
        })

def test_executes_gpu_process_tool(monkeypatch):

    def fake_processes():
        return {
            "count": 1,
            "processes": [
                {
                    "pid": 1234,
                    "name": "ollama.exe"
                }
            ]
        }

    monkeypatch.setattr(
        "gpu_tool_dispatcher.read_gpu_processes",
        fake_processes
    )

    result = execute_tool({
        "name": "read_gpu_processes",
        "arguments": {}
    })

    data = json.loads(result)

    assert data["count"] == 1
    assert data["processes"][0]["name"] == "ollama.exe"