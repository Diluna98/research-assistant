
from gpu_agent import run_gpu_agent
import json

def test_agent_answers_without_tools(monkeypatch, capsys):

    model_calls = 0

    def fake_chat(payload):
        nonlocal model_calls
        model_calls += 1

        return {
            "message": {
                "content": "GPU utilization measures GPU activity.",
                "tool_calls": []
            },
            "done_reason": "stop",
            "eval_count": 12
        }

    def forbidden_tool(function):
        raise AssertionError(
            "The agent should not execute a GPU tool"
        )

    monkeypatch.setattr(
        "gpu_agent.chat_or_error",
        fake_chat
    )

    monkeypatch.setattr(
        "gpu_agent.execute_tool",
        forbidden_tool
    )

    run_gpu_agent(
        "What does GPU utilization mean?"
    )

    output = capsys.readouterr().out

    assert model_calls == 1
    assert "GPU utilization measures GPU activity." in output





def test_agent_executes_requested_tool(monkeypatch, capsys):

    model_calls = 0
    tool_calls = 0

    def fake_chat(payload):
        nonlocal model_calls
        model_calls += 1

        if model_calls == 1:
            return {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_gpu_telemetry",
                                "arguments": {}
                            }
                        }
                    ]
                }
            }

        # Verify the second request includes tool evidence
        messages = payload["messages"]

        assert "tools" not in payload
        assert messages[-1]["role"] == "tool"
        assert messages[-1]["tool_name"] == "read_gpu_telemetry"

        measurements = json.loads(messages[-1]["content"])

        assert measurements["utilization_percent"] == 75

        return {
            "message": {
                "content": "Your GPU utilization is 75%."
            }
        }

    def fake_execute_tool(function):
        nonlocal tool_calls
        tool_calls += 1

        assert function["name"] == "read_gpu_telemetry"

        return json.dumps({
            "utilization_percent": 75
        })

    monkeypatch.setattr(
        "gpu_agent.chat_or_error",
        fake_chat
    )

    monkeypatch.setattr(
        "gpu_agent.execute_tool",
        fake_execute_tool
    )

    run_gpu_agent("What is my GPU utilization?")

    output = capsys.readouterr().out

    assert model_calls == 2
    assert tool_calls == 1
    assert "Your GPU utilization is 75%." in output

def test_agent_rejects_unauthorized_tool(monkeypatch, capsys):

    model_calls = 0

    def fake_chat(payload):
        nonlocal model_calls
        model_calls += 1

        return {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "delete_files",
                            "arguments": {}
                        }
                    }
                ]
            }
        }

    def forbidden_telemetry():
        raise AssertionError(
            "GPU telemetry should not execute"
        )

    monkeypatch.setattr(
        "gpu_agent.chat_or_error",
        fake_chat
    )

    monkeypatch.setattr(
        "gpu_tool_dispatcher.read_gpu_telemetry",
        forbidden_telemetry
    )

    run_gpu_agent("Delete my files")

    output = capsys.readouterr().out

    assert model_calls == 1
    assert "Rejected tool request" in output
    assert "Unsupported tool request" in output