from datetime import datetime
import json
from litellm import ModelResponse
import litellm
from loguru import logger


def process_litellm_callback(kwargs: dict, response: ModelResponse, start: datetime, end: datetime, event: str):
    assert isinstance(kwargs, dict), "kwargs must be a dictionary"
    assert isinstance(response, ModelResponse), "response must be a ModelResponse"
    assert isinstance(start, datetime), "start must be a datetime"
    assert isinstance(end, datetime), "end must be a datetime"
    assert isinstance(event, str), "event must be a string"

    # kwargs data
    model = kwargs.get("model")

    # NOTE: getting litellm_model caused an error that silently failed the processing of the callback :(
    # litellm_model = kwargs.get("litellm_params", {}).get("model_id", "")
    # assert model in litellm_model, f"model should be a subset of litellm_model: {model=}, {litellm_model=}"

    messages = kwargs.get("messages", [])
    cost = kwargs.get("response_cost", 0)
    cache_hit = kwargs.get("cache_hit", False)
    litellm_metadata = kwargs.get("litellm_params", {}).get("metadata", {})

    # cost check
    cost2 = litellm.completion_cost(completion_response=response)
    assert (cost - cost2) < 1e-6, "cost and cost2 should be the same"

    # response data
    response_dict = response.model_dump(mode="json")
    response_content = response_dict.get("choices", [{}])[0].get("message", {}).get("content", "")
    response_finish_reason = response_dict.get("choices", [{}])[0].get("finish_reason", "")

    response_valid_json = True
    try:
        json.loads(response_content)
    except Exception:
        response_valid_json = False

    # latency
    latency_ms = (end - start).total_seconds() * 1000

    return {
        "event": event,
        # "model": litellm_model,
        "model": model,
        "cost": cost,
        "latency_ms": latency_ms,
        "cache_hit": cache_hit,
        "messages": messages,
        "response_content": response_content,
        "response_finish_reason": response_finish_reason,
        "response_valid_json": response_valid_json,
        "response_dict": response_dict,
        "litellm_metadata": litellm_metadata,
    }


def on_litellm_success(kwargs, response, start, end):
    data = process_litellm_callback(kwargs, response, start, end, "success")
    logger.opt(depth=1).info(data)


def on_litellm_failure(kwargs, response, start, end):
    data = process_litellm_callback(kwargs, response, start, end, "failure")
    logger.opt(depth=1).info(data)
