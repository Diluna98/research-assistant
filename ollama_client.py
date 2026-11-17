
import requests
import logging
from config import load_config

logger = logging.getLogger("research_assistant.ollama")



def post_chat(payload):
    config = load_config()

    url = config["ollama_url"].rstrip("/") + "/api/chat"

    response = requests.post(
        url,
        json=payload,
        timeout=config["timeout"]
    )

    response.raise_for_status()

    return response.json()




def chat_or_error(payload):
    try:
        for attempt in range(2):
            try:
                return post_chat(payload)


            except requests.ConnectTimeout:
                if attempt == 0:
                    logger.warning(
                        "Retrying Ollama connection",
                        extra={
                            "event": "ollama_retry",
                            "attempt": attempt + 1,
                            "model": payload.get("model"),
                            "reason": "connection_timeout"
                        }
                    )
                else:
                    raise



    except requests.HTTPError as error:
        print("Ollama request failed.")

        if error.response is not None:
            print("HTTP status:", error.response.status_code)
            print("Server message:", error.response.text)
        else:
            print("Error:", error)


    except requests.ConnectionError as error:
        print("Could not connect to Ollama.")
        print("Details:", error)


    except requests.Timeout as error:
        print("Ollama request timed out.")
        print("Details:", error)


        return None