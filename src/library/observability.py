import json
import os
from statistics import mean
import threading

from gepa.core.result import GEPAResult
from loguru import logger


llm_calls_lock = threading.Lock()


def save_llm_calls(run_id: str, llm_calls: list[dict]):
    with llm_calls_lock:
        logger.info(f"Saving {len(llm_calls)} LLM calls to ./runs/{run_id}/llm_calls.json")
        os.makedirs(f"./runs/{run_id}", exist_ok=True)
        with open(f"./runs/{run_id}/llm_calls.json", "w") as f:
            json.dump(llm_calls, f, indent=2)


def summarize_llm_calls(run_id: str, script_path: str, llm_calls: list[dict], task_lm: str, reflection_lm: str) -> dict:
    total_lm_calls = len(llm_calls)
    # total_reflection_lm_calls = sum(1 for call in llm_calls if call["model"] == reflection_lm)  # call["model"] == "gpt-5"
    # total_task_lm_calls = sum(1 for call in llm_calls if call["model"] == task_lm)
    total_reflection_lm_calls = sum(
        1 for call in llm_calls if call["litellm_metadata"]["hidden_params"]["litellm_model_name"] == reflection_lm
    )  # call["litellm_metadata"]["hidden_params"]["litellm_model_name"] == "openai/gpt-5"
    total_task_lm_calls = sum(
        1 for call in llm_calls if call["litellm_metadata"]["hidden_params"]["litellm_model_name"] == task_lm
    )
    if sum([total_task_lm_calls, total_reflection_lm_calls]) != total_lm_calls:
        logger.error(
            f"Total LM calls ({total_lm_calls}) does not match the sum of task LM calls ({total_task_lm_calls}) and reflection LM calls ({total_reflection_lm_calls})"
        )

    total_cost_usd = sum(call["cost"] for call in llm_calls)
    # total_reflection_lm_cost_usd = sum(call["cost"] for call in llm_calls if call["model"] == reflection_lm)
    # total_task_lm_cost_usd = sum(call["cost"] for call in llm_calls if call["model"] == task_lm)
    total_reflection_lm_cost_usd = sum(
        call["cost"]
        for call in llm_calls
        if call["litellm_metadata"]["hidden_params"]["litellm_model_name"] == reflection_lm
    )
    total_task_lm_cost_usd = sum(
        call["cost"] for call in llm_calls if call["litellm_metadata"]["hidden_params"]["litellm_model_name"] == task_lm
    )
    if (total_task_lm_cost_usd + total_reflection_lm_cost_usd - total_cost_usd) > 1e-6:
        logger.error(
            f"Total cost USD ({total_cost_usd}) does not match the sum of task LM cost USD ({total_task_lm_cost_usd}) and reflection LM cost USD ({total_reflection_lm_cost_usd})"
        )

    mean_latency_ms = mean(call["latency_ms"] for call in llm_calls) if llm_calls else 0
    total_prompt_tokens = sum(
        call.get("response_dict", {}).get("usage", {}).get("prompt_tokens", 0) for call in llm_calls
    )
    total_completion_tokens = sum(
        call.get("response_dict", {}).get("usage", {}).get("completion_tokens", 0) for call in llm_calls
    )
    total_tokens = sum(call.get("response_dict", {}).get("usage", {}).get("total_tokens", 0) for call in llm_calls)

    summary = {
        "run_id": run_id,
        "script_path": script_path,
        "task_lm": task_lm,
        "reflection_lm": reflection_lm,
        "total_lm_calls": total_lm_calls,
        "total_task_lm_calls": total_task_lm_calls,
        "total_reflection_lm_calls": total_reflection_lm_calls,
        "total_cost_usd": total_cost_usd,
        "total_task_lm_cost_usd": total_task_lm_cost_usd,
        "total_reflection_lm_cost_usd": total_reflection_lm_cost_usd,
        "total_task_lm_cost_usd_percentage": (total_task_lm_cost_usd / total_cost_usd) * 100
        if total_cost_usd > 0
        else 0,
        "total_reflection_lm_cost_usd_percentage": (total_reflection_lm_cost_usd / total_cost_usd) * 100
        if total_cost_usd > 0
        else 0,
        "mean_latency_ms": mean_latency_ms,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
    }

    os.makedirs(f"./runs/{run_id}", exist_ok=True)
    with open(f"./runs/{run_id}/llm_calls_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def summarize_run(
    run_id: str,
    script_path: str,
    seed_prompt: dict,
    trainset: list,
    valset: list,
    testset: list,
    max_metric_calls: int,
    task_lm: str,
    reflection_lm: str,
    gepa_result: GEPAResult,
    summary_llm_calls: dict,
) -> dict:
    candidates = gepa_result.candidates
    initial_seed_system_prompt = seed_prompt["system_prompt"]
    best_candidate_system_prompt = gepa_result.best_candidate["system_prompt"]
    val_aggregate_scores = gepa_result.val_aggregate_scores
    if len(val_aggregate_scores) != len(candidates):
        logger.error(
            f"Length of val_aggregate_scores ({len(val_aggregate_scores)}) does not match the length of candidates ({len(candidates)})"
        )

    gepa_result_dict = gepa_result.to_dict()

    summary = {
        "run_id": run_id,
        "script_path": script_path,
        "max_metric_calls": max_metric_calls,
        "task_lm": task_lm,
        "reflection_lm": reflection_lm,
        "len_trainset": len(trainset),
        "len_valset": len(valset),
        "len_testset": len(testset),
        "total_lm_calls": summary_llm_calls["total_lm_calls"],
        "total_task_lm_calls": summary_llm_calls["total_task_lm_calls"],
        "total_reflection_lm_calls": summary_llm_calls["total_reflection_lm_calls"],
        "total_cost_usd": summary_llm_calls["total_cost_usd"],
        "total_task_lm_cost_usd": summary_llm_calls["total_task_lm_cost_usd"],
        "total_reflection_lm_cost_usd": summary_llm_calls["total_reflection_lm_cost_usd"],
        "gepa_seed_system_prompt": initial_seed_system_prompt,
        "gepa_best_system_prompt": best_candidate_system_prompt,
        "len_gepa_seed_system_prompt": len(initial_seed_system_prompt),
        "len_gepa_best_system_prompt": len(best_candidate_system_prompt),
        "seed_and_best_different": initial_seed_system_prompt != best_candidate_system_prompt,
        "len_gepa_candidates": len(candidates),
        "gepa_candidates": candidates,
        "gepa_scores": val_aggregate_scores,
        "gepa_result_dict": gepa_result_dict,
    }

    os.makedirs(f"./runs/{run_id}", exist_ok=True)
    with open(f"./runs/{run_id}/run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary
