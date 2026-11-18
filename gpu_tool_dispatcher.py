
import json

from gpu_telemetry import read_gpu_telemetry
from gpu_processes import read_gpu_processes


def execute_tool(function):

    name = function.get("name")
    arguments = function.get("arguments", {})

    allowed_tools = {
        "read_gpu_telemetry": read_gpu_telemetry,
        "read_gpu_processes": read_gpu_processes
    }

    if name not in allowed_tools or arguments != {}:
        raise ValueError("Unsupported tool request")

    try:
        result = allowed_tools[name]()
        return json.dumps(result)

    except RuntimeError as error:
        return json.dumps({
            "error": str(error)
        })
