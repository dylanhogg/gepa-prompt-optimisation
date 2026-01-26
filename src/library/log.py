import json
import os
import sys
from gepa.logging.logger import LoggerProtocol
from loguru import logger


def configure_loguru(remove_existing: bool = True, log_dir: str = "./log") -> None:
    from datetime import datetime

    if remove_existing:
        logger.remove()

    # Create datetime-stamped log file for this session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(log_dir, f"{timestamp}.log")

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    logger.add(sys.stderr, level=os.getenv("LOG_STDERR_LEVEL", "INFO"))
    logger.add(
        logfile,
        level=os.getenv("LOG_FILE_LEVEL", "DEBUG"),
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


def log_results(gepa_result, seed_prompt):
    logger.info("--------------------------------")
    logger.info(f"{gepa_result=}\n\n")

    candidates = gepa_result.candidates
    initial_seed_system_prompt = seed_prompt["system_prompt"]
    best_candidate_system_prompt = gepa_result.best_candidate["system_prompt"]

    logger.info(f"Number of candidates: {len(candidates)}")
    for i, candidate in enumerate(candidates):
        logger.info(f"Candidate {i + 1}:")
        logger.info(f"{json.dumps(candidate, indent=4)}\n\n")
    logger.info("--------------------------------")
    logger.info(f"{initial_seed_system_prompt=}\n\n")
    logger.info(f"{best_candidate_system_prompt=}\n\n")
