
import subprocess
from pathlib import PureWindowsPath


def read_gpu_processes():

    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name",
        "--format=csv,noheader"
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired
    ) as error:
        raise RuntimeError(
            "Could not retrieve GPU process information"
        ) from error

    processes = []

    for line in result.stdout.splitlines():

        if not line.strip():
            continue

        try:
            pid_text, process_path = line.split(",", 1)
            pid = int(pid_text.strip())

        except ValueError as error:
            raise RuntimeError(
                "Invalid GPU process information"
            ) from error

        process_path = process_path.strip()

        if process_path == "[Insufficient Permissions]":
            process_name = None
        else:
            process_name = PureWindowsPath(process_path).name

        processes.append({
            "pid": pid,
            "name": process_name
        })

    return {
        "count": len(processes),
        "processes": processes
    }
