
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


def test_executes_paper_search(monkeypatch):

    def fake_search(question, top_k):
        assert question == "Explain Equation 11"
        assert top_k == 3

        return {
            "question": question,
            "results": [
                {
                    "page": 4,
                    "chunk_id": 21,
                    "text": "Example research evidence."
                }
            ]
        }

    monkeypatch.setattr(
        "gpu_tool_dispatcher.search_paper",
        fake_search
    )

    result = execute_tool({
        "name": "search_paper",
        "arguments": {
            "question": "Explain Equation 11",
            "top_k": 3
        }
    })

    data = json.loads(result)

    assert data["results"][0]["chunk_id"] == 21
    assert data["results"][0]["page"] == 4


def test_executes_workload(monkeypatch):

    calls = 0

    def fake_workload():
        nonlocal calls
        calls += 1

        return {
            "model": "qwen3:4b-instruct",
            "wall_time_s": 4.7,
            "tokens": 200,
            "tokens_per_second": 43.0
        }

    monkeypatch.setattr(
        "gpu_tool_dispatcher.run_workload",
        fake_workload
    )

    result = execute_tool({
        "name": "run_workload",
        "arguments": {}
    })

    data = json.loads(result)

    assert calls == 1
    assert data["tokens"] == 200
    assert data["tokens_per_second"] == 43.0


def test_rejects_workload_arguments():

    with pytest.raises(
        ValueError,
        match="Unsupported tool request"
    ):
        execute_tool({
            "name": "run_workload",
            "arguments": {
                "model": "unknown-model"
            }
        })


def test_executes_workload_evaluation(monkeypatch):

    calls = 0

    def fake_evaluation():
        nonlocal calls
        calls += 1

        return {
            "benchmark": {
                "model": "qwen3:4b-instruct",
                "average_speed": 59.5,
                "average_latency": 3.47
            },
            "comparison": {
                "status": "within_tolerance",
                "speed_change_percent": -0.5,
                "latency_change_percent": 0.3
            }
        }

    monkeypatch.setattr(
        "gpu_tool_dispatcher.evaluate_workload",
        fake_evaluation
    )

    result = execute_tool({
        "name": "evaluate_workload",
        "arguments": {}
    })

    data = json.loads(result)

    assert calls == 1
    assert data["comparison"]["status"] == "within_tolerance"
    assert data["benchmark"]["average_speed"] == 59.5


def test_rejects_evaluation_arguments():

    with pytest.raises(
        ValueError,
        match="Unsupported tool request"
    ):
        execute_tool({
            "name": "evaluate_workload",
            "arguments": {
                "runs": 100
            }
        })


def test_executes_workload_diagnostics(monkeypatch):

    calls = 0

    def fake_diagnosis():
        nonlocal calls
        calls += 1

        return {
            "evaluation": {
                "comparison": {
                    "status": "possible_regression"
                }
            },
            "diagnostics": {
                "gpu_telemetry": {
                    "utilization_percent": 95
                }
            }
        }

    monkeypatch.setattr(
        "gpu_tool_dispatcher.diagnose_workload",
        fake_diagnosis
    )

    result = execute_tool({
        "name": "diagnose_workload",
        "arguments": {}
    })

    data = json.loads(result)

    assert calls == 1
    assert (
        data["evaluation"]["comparison"]["status"]
        == "possible_regression"
    )
    assert (
        data["diagnostics"]["gpu_telemetry"]
        ["utilization_percent"]
        == 95
    )


def test_rejects_diagnostic_arguments():

    with pytest.raises(ValueError):
        execute_tool({
            "name": "diagnose_workload",
            "arguments": {
                "runs": 100
            }
        })
