
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


def test_agent_uses_gpu_process_tool(monkeypatch, capsys):

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
                                "name": "read_gpu_processes",
                                "arguments": {}
                            }
                        }
                    ]
                }
            }

        messages = payload["messages"]

        assert messages[-1]["role"] == "tool"
        assert messages[-1]["tool_name"] == "read_gpu_processes"

        result = json.loads(messages[-1]["content"])

        assert result["count"] == 1
        assert result["processes"][0]["name"] == "ollama.exe"

        return {
            "message": {
                "content": "Ollama is reported as using the GPU."
            }
        }

    def fake_execute_tool(function):
        nonlocal tool_calls
        tool_calls += 1

        assert function["name"] == "read_gpu_processes"
        assert function["arguments"] == {}

        return json.dumps({
            "count": 1,
            "processes": [
                {
                    "pid": 1234,
                    "name": "ollama.exe"
                }
            ]
        })

    monkeypatch.setattr(
        "gpu_agent.chat_or_error",
        fake_chat
    )

    monkeypatch.setattr(
        "gpu_agent.execute_tool",
        fake_execute_tool
    )

    run_gpu_agent(
        "Which applications are using my GPU?"
    )

    output = capsys.readouterr().out

    assert model_calls == 2
    assert tool_calls == 1
    assert "Ollama is reported as using the GPU." in output


def test_agent_executes_two_tools(monkeypatch, capsys):

    model_calls = 0
    executed_tools = []

    def fake_chat(payload):
        nonlocal model_calls
        model_calls += 1

        if model_calls == 1:
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_gpu_telemetry",
                                "arguments": {}
                            }
                        },
                        {
                            "function": {
                                "name": "read_gpu_processes",
                                "arguments": {}
                            }
                        }
                    ]
                }
            }

        messages = payload["messages"]

        assert "tools" not in payload

        assert messages[-2]["role"] == "tool"
        assert messages[-1]["role"] == "tool"

        assert messages[-2]["tool_name"] == "read_gpu_telemetry"
        assert messages[-1]["tool_name"] == "read_gpu_processes"

        telemetry = json.loads(messages[-2]["content"])
        processes = json.loads(messages[-1]["content"])

        assert telemetry["utilization_percent"] == 75
        assert processes["count"] == 1

        return {
            "message": {
                "content": "GPU is at 75%. Ollama is using the GPU."
            }
        }

    def fake_execute_tool(function):

        name = function["name"]
        executed_tools.append(name)

        if name == "read_gpu_telemetry":
            return json.dumps({
                "utilization_percent": 75
            })

        if name == "read_gpu_processes":
            return json.dumps({
                "count": 1,
                "processes": [
                    {"pid": 1234, "name": "ollama.exe"}
                ]
            })

        raise AssertionError("Unexpected tool")

    monkeypatch.setattr(
        "gpu_agent.chat_or_error",
        fake_chat
    )

    monkeypatch.setattr(
        "gpu_agent.execute_tool",
        fake_execute_tool
    )

    run_gpu_agent(
        "What is my GPU utilization and which processes use it?"
    )

    output = capsys.readouterr().out

    assert model_calls == 2

    assert executed_tools == [
        "read_gpu_telemetry",
        "read_gpu_processes"
    ]

    assert "GPU is at 75%" in output
    assert "Ollama is using the GPU" in output