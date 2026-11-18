
from workload_diagnostics import diagnose_workload


def test_investigates_regression(monkeypatch):

    calls = []

    monkeypatch.setattr(
        "workload_diagnostics.evaluate_workload",
        lambda: {
            "comparison": {
                "status": "possible_regression"
            }
        }
    )

    def fake_telemetry():
        calls.append("telemetry")
        return {"utilization_percent": 95}

    def fake_processes():
        calls.append("processes")
        return {"count": 2}

    monkeypatch.setattr(
        "workload_diagnostics.read_gpu_telemetry",
        fake_telemetry
    )

    monkeypatch.setattr(
        "workload_diagnostics.read_gpu_processes",
        fake_processes
    )

    result = diagnose_workload()

    assert calls == ["telemetry", "processes"]

    assert (
        result["diagnostics"]
        ["gpu_telemetry"]["utilization_percent"]
        == 95
    )


def test_skips_diagnostics_without_regression(monkeypatch):

    def fake_evaluation():
        return {
            "comparison": {
                "status": "within_tolerance"
            }
        }

    def forbidden_diagnostic():
        raise AssertionError(
            "Diagnostics should not execute"
        )

    monkeypatch.setattr(
        "workload_diagnostics.evaluate_workload",
        fake_evaluation
    )

    monkeypatch.setattr(
        "workload_diagnostics.read_gpu_telemetry",
        forbidden_diagnostic
    )

    monkeypatch.setattr(
        "workload_diagnostics.read_gpu_processes",
        forbidden_diagnostic
    )

    result = diagnose_workload()

    assert result["diagnostics"] is None

    assert (
        result["evaluation"]["comparison"]["status"]
        == "within_tolerance"
    )