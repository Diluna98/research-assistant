
import json
from gpu_telemetry import read_gpu_telemetry
from gpu_processes import read_gpu_processes
from research_tools import search_paper
from workload_runner import run_workload
from workload_evaluator import evaluate_workload
from workload_diagnostics import diagnose_workload

def execute_tool(function):

    name = function.get("name")
    arguments = function.get("arguments", {})

    if not isinstance(arguments, dict):
        raise ValueError("Unsupported tool request")

    allowed_gpu_tools = {
        "read_gpu_telemetry": read_gpu_telemetry,
        "read_gpu_processes": read_gpu_processes
    }

    if name in allowed_gpu_tools:

        if arguments != {}:
            raise ValueError("Unsupported tool request")

        try:
            result = allowed_gpu_tools[name]()
            return json.dumps(result)

        except RuntimeError as error:
            return json.dumps({"error": str(error)})

    if name == "search_paper":

        if set(arguments) - {"question", "top_k"}:
            raise ValueError("Unsupported tool request")

        question = arguments.get("question")
        top_k = arguments.get("top_k", 5)

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Unsupported tool request")

        if type(top_k) is not int or not 1 <= top_k <= 10:
            raise ValueError("Unsupported tool request")

        result = search_paper(question, top_k=top_k)

        return json.dumps(result)

    if name == "run_workload":

        if arguments != {}:
            raise ValueError("Unsupported tool request")

        result = run_workload()

        return json.dumps(result)


    if name == "evaluate_workload":

        if arguments != {}:
            raise ValueError("Unsupported tool request")

        result = evaluate_workload()

        return json.dumps(result)


    if name == "diagnose_workload":

        if arguments != {}:
            raise ValueError("Unsupported tool request")

        result = diagnose_workload()

        return json.dumps(result)


    raise ValueError("Unsupported tool request")
