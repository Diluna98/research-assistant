
from workload_evaluator import evaluate_workload


def test_evaluate_workload(monkeypatch):

    benchmark_calls = 0
    comparison_calls = 0
    logged_events = []

    benchmark_result = {
        "model": "qwen3:4b-instruct",
        "runs": 5,
        "average_speed": 59.5,
        "average_latency": 3.47
    }

    comparison_result = {
        "status": "within_tolerance",
        "speed_change_percent": -0.5,
        "latency_change_percent": 0.3
    }

    def fake_benchmark(runs):
        nonlocal benchmark_calls
        benchmark_calls += 1

        assert runs == 5

        return benchmark_result

    def fake_comparison(summary):
        nonlocal comparison_calls
        comparison_calls += 1

        assert summary == benchmark_result

        return comparison_result

    class FakeLogger:
        def info(self, message, extra):
            logged_events.append({
                "message": message,
                "extra": extra
            })

    monkeypatch.setattr(
        "workload_evaluator.benchmark_workload",
        fake_benchmark
    )

    monkeypatch.setattr(
        "workload_evaluator.compare_with_baseline",
        fake_comparison
    )

    monkeypatch.setattr(
        "workload_evaluator.configure_logging",
        lambda: FakeLogger()
    )

    result = evaluate_workload()

    assert benchmark_calls == 1
    assert comparison_calls == 1

    assert result["benchmark"] == benchmark_result
    assert result["comparison"] == comparison_result

    assert len(logged_events) == 1

    assert (
        logged_events[0]["extra"]["event"]
        == "workload_evaluation"
    )