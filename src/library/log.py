import json
import os
import sys
from gepa.logging.logger import LoggerProtocol
from loguru import logger


def configure_loguru(remove_existing: bool = True, logfile: str = "./log/app.log") -> None:
    if remove_existing:
        logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_STDERR_LEVEL", "INFO"))
    logger.add(
        logfile,
        level=os.getenv("LOG_FILE_LEVEL", "DEBUG"),
        rotation=os.getenv("LOG_FILE_ROTATION", "00:00"),
    )


class CustomGepaLogger(LoggerProtocol):
    def log(self, message: str):
        try:
            message_dict = json.loads(message)
            logger.opt(depth=1).info(f"{json.dumps(message_dict, indent=4)}")
        except Exception:
            logger.opt(depth=1).info(f"{message}")


def on_litellm_success(kwargs, response, start, end):
    logger.opt(depth=1).info(
        {
            "model": kwargs.get("model"),
            "messages": kwargs.get("messages"),
            "latency_ms": (end - start) * 1000,
            "response": response,
        }
    )


def on_litellm_failure(kwargs, response, start, end):
    logger.opt(depth=1).error(
        {
            "model": kwargs.get("model"),
            "messages": kwargs.get("messages"),
            "latency_ms": (end - start) * 1000,
            "error": response,
        }
    )
