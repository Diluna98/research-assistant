
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


gpu_process_tool = {
    "type": "function",
    "function": {
        "name": "read_gpu_processes",
        "description": (
            "List processes currently reported by NVIDIA "
            "as using the GPU. Returns process IDs and names. "
            "Does not report memory usage per process."
        ),
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
}


def run_gpu_agent(question: str):
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty")
    payload = {
        "model": load_config()["model"],
        "messages": [
            {
                "role": "system",
                "content": (
                            "You are a GPU diagnostic assistant. "
                            "Use read_gpu_telemetry for current GPU utilization, "
                            "temperature, and total GPU memory measurements. "
                            "Use read_gpu_processes to identify applications "
                            "reported as using the GPU. "
                            "Answer conceptual questions without tools. "
                            "Never invent measurements or process information. "
                            "Individual process GPU memory usage is unavailable. "
                            "Distinguish observed measurements from assumptions. "
                            "Be concise."
                        )

            },
            {
                "role": "user",
                "content": question
            }
        ],
        "tools": [gpu_tool, gpu_process_tool],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": 512
        }
    }



    response = chat_or_error(payload)

    if response is not None:
        assistant_message = response["message"]
        tool_calls = assistant_message.get("tool_calls", [])


    if len(tool_calls) == 0:

        print("\nFinal assistant answer:")
        print(assistant_message.get("content", ""))

    else:
        functions = [
            call["function"]
            for call in tool_calls
        ]

        allowed_names = {
            "read_gpu_telemetry",
            "read_gpu_processes"
        }

        names = [
            function.get("name")
            for function in functions
        ]

        invalid_request = (
            len(functions) > 2
            or len(set(names)) != len(names)
            or any(
                function.get("name") not in allowed_names
                or function.get("arguments", {}) != {}
                for function in functions
            )
        )

        if invalid_request:
            print("Rejected tool request: Unsupported tool request")
            return

        messages = payload["messages"] + [assistant_message]

        for function in functions:

            print("Selected tool:", function["name"])

            try:
                tool_result = execute_tool(function)

            except ValueError as error:
                print("Rejected tool request:", error)
                return

            print("GPU tool result:")
            print(tool_result)

            messages.append({
                "role": "tool",
                "tool_name": function["name"],
                "content": tool_result
            })

        final_payload = {
            **payload,
            "messages": messages
        }

        final_payload.pop("tools", None)

        final_response = chat_or_error(final_payload)

        if final_response is not None:

            final_message = final_response["message"]

            if final_message.get("tool_calls"):
                print("Further tool requests are not supported yet.")
                return

            print("\nFinal assistant answer:")
            print(final_message.get("content", ""))



    if response is not None:
        message = response["message"]

if __name__ == "__main__":
    question = input("Ask the GPU agent: ")
    run_gpu_agent(question)