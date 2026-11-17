
import subprocess


def read_gpu_telemetry():

    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits"
    ]


    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )

    except FileNotFoundError as error:
        raise RuntimeError(
            "nvidia-smi is not available"
        ) from error


    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "nvidia-smi command failed"
        ) from error

    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "nvidia-smi timed out"
        ) from error



    values = result.stdout.strip().split(",")

    name, utilization, used, total, temperature = [
        value.strip() for value in values
    ]

    return {
        "name": name,
        "utilization_percent": int(utilization),
        "memory_used_mib": int(used),
        "memory_total_mib": int(total),
        "temperature_c": int(temperature)
    }
