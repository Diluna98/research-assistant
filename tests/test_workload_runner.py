
from workload_runner import run_workload


def test_workload_execution(monkeypatch):

    def fake_config():
        return {
            "model": "qwen3:4b-instruct"
        }

    def fake_chat(payload):

        assert payload["model"] == "qwen3:4b-instruct"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0
        assert payload["options"]["num_predict"] == 200

        return {
            "total_duration": 5_000_000_000,
            "eval_duration": 4_000_000_000,
            "eval_count": 200,
            "done_reason": "stop"
        }

    monkeypatch.setattr(
        "workload_runner.load_config",
        fake_config
    )

    monkeypatch.setattr(
        "workload_runner.post_chat",
        fake_chat
    )

    result = run_workload()

    assert result["model"] == "qwen3:4b-instruct"
    assert result["total_duration_s"] == 5.0
    assert result["generation_time_s"] == 4.0
    assert result["tokens"] == 200
    assert result["tokens_per_second"] == 50.0
    assert result["stop_reason"] == "stop"
    assert result["wall_time_s"] >= 0


def test_zero_generation_duration(monkeypatch):

    monkeypatch.setattr(
        "workload_runner.load_config",
        lambda: {"model": "qwen3:4b-instruct"}
    )

    monkeypatch.setattr(
        "workload_runner.post_chat",
        lambda payload: {
            "total_duration": 100_000_000,
            "eval_duration": 0,
            "eval_count": 0,
            "done_reason": "stop"
        }
    )

    result = run_workload()

    assert result["tokens_per_second"] == 0