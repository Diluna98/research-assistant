
from workload_evaluator import evaluate_workload
from gpu_telemetry import read_gpu_telemetry
from gpu_processes import read_gpu_processes


def diagnose_workload():

    evaluation = evaluate_workload()

    status = evaluation["comparison"]["status"]

    if status != "possible_regression":
        return {
            "evaluation": evaluation,
            "diagnostics": None
        }

    diagnostics = {}

    for name, diagnostic in {
        "gpu_telemetry": read_gpu_telemetry,
        "gpu_processes": read_gpu_processes
    }.items():

        try:
            diagnostics[name] = diagnostic()

        except RuntimeError as error:
            diagnostics[name] = {
                "error": str(error)
            }

    return {
        "evaluation": evaluation,
        "diagnostics": diagnostics
    }