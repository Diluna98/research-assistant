
import os
from pathlib import Path
from dotenv import load_dotenv



def load_config():
    load_dotenv(
        dotenv_path=Path(__file__).with_name(".env"),
        override=False
    )

    return {
        "model": os.getenv(
            "RA_MODEL",
            "qwen3:4b-instruct"
        ),
        "ollama_url": os.getenv(
            "RA_OLLAMA_URL",
            "http://127.0.0.1:11434"
        ),
        "timeout": int(os.getenv(
            "RA_TIMEOUT",
            "180"
        ))
    }
