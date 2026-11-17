
import json

from gpu_telemetry import read_gpu_telemetry


def execute_tool(function):

    name = function.get("name")
    arguments = function.get("arguments", {})

    if name != "read_gpu_telemetry" or arguments != {}:
        raise ValueError("Unsupported tool request")

    try:
        result = read_gpu_telemetry()
        return json.dumps(result)

    except RuntimeError as error:
        return json.dumps({
            "error": str(error)
        })
