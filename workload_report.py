
def format_diagnosis_report(result):

    evaluation = result["evaluation"]

    benchmark = evaluation["benchmark"]
    comparison = evaluation["comparison"]

    status = comparison["status"]

    lines = [
        "GPU Workload Evaluation",
        "",
        f"Model: {benchmark['model']}",
        f"Runs: {benchmark['runs']}",
        f"Average speed: "
        f"{benchmark['average_speed']:.2f} tokens/s",
        f"Average latency: "
        f"{benchmark['average_latency']:.3f} seconds",
        "",
        "Baseline comparison:",
        f"Speed change: "
        f"{comparison['speed_change_percent']:+.2f}%",
        f"Latency change: "
        f"{comparison['latency_change_percent']:+.2f}%",
        f"Status: {status}"
    ]

    if status == "within_tolerance":
        lines.append(
            "No regression detected under the configured "
            "tolerance. Additional diagnostics were skipped."
        )

    elif status == "possible_regression":
        lines.append(
            "Possible performance regression detected."
        )

        diagnostics = result.get("diagnostics")

        if diagnostics is not None:
            lines.append(
                f"Diagnostics: {diagnostics}"
            )

    return "\n".join(lines)