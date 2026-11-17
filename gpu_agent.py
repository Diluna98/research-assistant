
from config import load_config
from ollama_client import chat_or_error
import json
from gpu_tool_dispatcher import execute_tool

gpu_tool = {
    "type": "function",
    "function": {
        "name": "read_gpu_telemetry",
        "description": (
            "Get current NVIDIA GPU utilization, "
            "memory usage, and temperature."
        ),
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
}


payload = {
    "model": load_config()["model"],
    "messages": [
        {
            "role": "system",
            "content": (
                "You are a GPU diagnostic assistant. "
                "Use the available tool when current "
                "GPU measurements are required. "
                "Never invent measurements."
            )
        },
        {
            "role": "user",
            "content": "What is my current GPU utilization?"
        }
    ],
    "tools": [gpu_tool],
    "stream": False,
    "think": False,
    "options": {
        "temperature": 0,
        "num_predict": 256
    }
}



response = chat_or_error(payload)

if response is not None:
    assistant_message = response["message"]
    tool_calls = assistant_message.get("tool_calls", [])

    if len(tool_calls) != 1:
        print("Expected exactly one tool request.")
        print("Assistant:", assistant_message.get("content", ""))

    else:

        function = tool_calls[0]["function"]

        try:
            tool_result = execute_tool(function)

        except ValueError as error:
            print("Rejected tool request:", error)

        else:
            print("GPU tool result:")
            print(tool_result)

            messages = payload["messages"] + [
                assistant_message,
                {
                    "role": "tool",
                    "tool_name": function["name"],
                    "content": tool_result
                }
            ]

            final_payload = {
                **payload,
                "messages": messages
            }

            final_payload.pop("tools", None)

            final_response = chat_or_error(final_payload)

            if final_response is not None:
                print("\nFinal assistant answer:")
                print(
                    final_response["message"].get("content", "")
                )



if response is not None:
    message = response["message"]