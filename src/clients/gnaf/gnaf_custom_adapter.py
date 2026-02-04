import json
import os
import re
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple, Protocol, TypedDict, cast

import litellm
import typer
from dotenv import load_dotenv
from gepa.api import optimize
from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from loguru import logger
from stickler.structured_object_evaluator.models.structured_model import StructuredModel

from clients.example.gnaf_common import get_seed_prompt, init_dataset_default_adapter, log_results
from library import llm, log, observability

assert load_dotenv(), "Failed to load .env file"
assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"

app = typer.Typer(pretty_exceptions_enable=False, pretty_exceptions_show_locals=True)
run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"

llm_calls: list[dict[str, Any]] = []
llm_calls_lock = threading.Lock()


def on_litellm_input(kwargs, input, start, end):
    data = llm.process_litellm_callback(kwargs, input, start, end, "input")
    logger.opt(depth=1).warning(data)
    with llm_calls_lock:
        llm_calls.append(data)
    observability.save_llm_calls(run_id, llm_calls)


def on_litellm_success(kwargs, response, start, end):
    data = llm.process_litellm_callback(kwargs, response, start, end, "success")
    logger.opt(depth=1).warning(data)
    with llm_calls_lock:
        llm_calls.append(data)
    observability.save_llm_calls(run_id, llm_calls)


def on_litellm_failure(kwargs, response, start, end):
    data = llm.process_litellm_callback(kwargs, response, start, end, "failure")
    logger.opt(depth=1).warning(data)
    with llm_calls_lock:
        llm_calls.append(data)
    observability.save_llm_calls(run_id, llm_calls)


litellm.input_callback = [on_litellm_input]
litellm.success_callback = [on_litellm_success]
litellm.failure_callback = [on_litellm_failure]


"""
Custom GEPA adapter + evaluator for GNAF using Stickler structured evaluation.
"""

GNAF_FIELDS = [
    "building_name",
    "flat_number",
    "flat_number_prefix",
    "flat_number_suffix",
    "flat_type",
    "level_number",
    "level_number_prefix",
    "level_number_suffix",
    "level_type",
    "locality_name",
    "lot_number",
    "lot_number_prefix",
    "lot_number_suffix",
    "number_first",
    "number_first_prefix",
    "number_first_suffix",
    "number_last",
    "number_last_prefix",
    "number_last_suffix",
    "postcode",
    "state_abbreviation",
    "street_name",
    "street_suffix_code",
    "street_type_code",
]


class DefaultDataInst(TypedDict):
    input: str
    additional_context: dict[str, str]
    answer: str


class EvaluationResult(NamedTuple):
    score: float
    feedback: str
    objective_scores: dict[str, float] | None = None


class DefaultTrajectory(TypedDict):
    data: DefaultDataInst
    full_assistant_response: str
    feedback: str


class DefaultRolloutOutput(TypedDict):
    full_assistant_response: str


DefaultReflectiveRecord = TypedDict(
    "DefaultReflectiveRecord",
    {
        "Inputs": str,
        "Generated Outputs": str,
        "Feedback": str,
    },
)


class ChatMessage(TypedDict):
    role: str
    content: str


class ChatCompletionCallable(Protocol):
    """Protocol for chat completion callables (duck typing for custom model wrappers)."""

    def __call__(self, messages: Sequence[ChatMessage]) -> str: ...


class Evaluator(Protocol):
    def __call__(self, data: DefaultDataInst, response: str) -> EvaluationResult: ...


def _parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    if not text:
        return None, "Empty response"

    def _try_load(candidate: str) -> tuple[dict[str, Any] | None, str | None]:
        try:
            obj = json.loads(candidate)
        except Exception as exc:  # noqa: BLE001
            return None, f"Invalid JSON: {exc}"
        if not isinstance(obj, dict):
            return None, "JSON must be an object"
        return obj, None

    obj, err = _try_load(text)
    if obj is not None:
        return obj, None

    for match in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL):
        obj, err = _try_load(match)
        if obj is not None:
            return obj, None

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        obj, err = _try_load(text[start : end + 1])
        if obj is not None:
            return obj, None

    return None, err or "Unable to parse JSON object"


def _normalize_gnaf_dict(raw: Mapping[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for field in GNAF_FIELDS:
        value = raw.get(field)
        if value is None:
            normalized[field] = ""
        elif isinstance(value, (int, float)):
            normalized[field] = str(value)
        else:
            normalized[field] = str(value)
    return normalized


def _build_stickler_config(match_threshold: float, string_threshold: float) -> dict[str, Any]:
    return {
        "model_name": "GNAFAddress",
        "match_threshold": match_threshold,
        "fields": {
            field: {
                "type": "str",
                "comparator": "LevenshteinComparator",
                "threshold": string_threshold,
                "weight": 1.0,
                "default": "",
            }
            for field in GNAF_FIELDS
        },
    }


class GNAFSticklerEvaluator:
    def __init__(
        self,
        failure_score: float = 0.0,
        match_threshold: float = 0.8,
        string_threshold: float = 0.9,
    ):
        self.failure_score = failure_score
        self.model_cls = StructuredModel.model_from_json(
            _build_stickler_config(match_threshold=match_threshold, string_threshold=string_threshold)
        )

    def __call__(self, data: DefaultDataInst, response: str) -> EvaluationResult:
        expected_raw, expected_err = _parse_json_object(data["answer"])
        if expected_raw is None:
            feedback = f"Ground-truth JSON is invalid: {expected_err}"
            return EvaluationResult(score=self.failure_score, feedback=feedback, objective_scores=None)

        predicted_raw, predicted_err = _parse_json_object(response)
        if predicted_raw is None:
            feedback = (
                f"Response is not valid JSON: {predicted_err}. "
                "Return only a JSON object with extracted GNAF fields and no extra text."
            )
            return EvaluationResult(score=self.failure_score, feedback=feedback, objective_scores=None)

        expected = _normalize_gnaf_dict(expected_raw)
        predicted = _normalize_gnaf_dict(predicted_raw)

        expected_model = self.model_cls(**expected)  # type: ignore[reportCallIssue]
        predicted_model = self.model_cls(**predicted)  # type: ignore[reportCallIssue]

        result = expected_model.compare_with(predicted_model)
        score = float(result.get("overall_score", self.failure_score))

        extra_keys = sorted(set(predicted_raw) - set(GNAF_FIELDS))
        if extra_keys:
            score = max(self.failure_score, score - min(0.2, 0.02 * len(extra_keys)))

        mismatches: list[str] = []
        for field in GNAF_FIELDS:
            expected_value = str(expected_raw.get(field, ""))
            predicted_value = str(predicted_raw.get(field, ""))
            if expected_value != predicted_value:
                mismatches.append(f"{field} expected={expected_value!r} predicted={predicted_value!r}")

        feedback_parts = [f"Structured match score {score:.2f} (Stickler)."]
        missing_keys = sorted(set(expected_raw) - set(predicted_raw))
        if missing_keys:
            feedback_parts.append(f"Missing keys: {', '.join(missing_keys)}.")
        if extra_keys:
            feedback_parts.append(f"Unexpected keys: {', '.join(extra_keys)}.")
        if mismatches:
            feedback_parts.append("Mismatched fields: " + "; ".join(mismatches[:6]) + ".")
        if not mismatches and not extra_keys:
            feedback_parts.append("Response matches expected JSON.")

        return EvaluationResult(score=score, feedback=" ".join(feedback_parts), objective_scores=None)


class CustomGEPAAdapter(GEPAAdapter[DefaultDataInst, DefaultTrajectory, DefaultRolloutOutput]):
    def __init__(
        self,
        model: str | ChatCompletionCallable,
        evaluator: Evaluator | None = None,
        max_litellm_workers: int = 10,
        litellm_batch_completion_kwargs: dict[str, Any] | None = None,
    ):
        if isinstance(model, str):
            import litellm

            self.litellm = litellm
        self.model = model
        self.evaluator = evaluator or GNAFSticklerEvaluator()
        self.max_litellm_workers = max_litellm_workers
        self.litellm_batch_completion_kwargs = litellm_batch_completion_kwargs or {}

    def evaluate(
        self,
        batch: list[DefaultDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[DefaultTrajectory, DefaultRolloutOutput]:
        outputs: list[DefaultRolloutOutput] = []
        scores: list[float] = []
        objective_scores: list[dict[str, float] | None] = []
        trajectories: list[DefaultTrajectory] | None = [] if capture_traces else None

        system_content = next(iter(candidate.values()))
        litellm_requests = []

        for data in batch:
            user_content = f"{data['input']}"
            messages: list[ChatMessage] = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
            litellm_requests.append(messages)

        if isinstance(self.model, str):
            responses = [
                resp.choices[0].message.content.strip()
                for resp in self.litellm.batch_completion(
                    model=self.model,
                    messages=litellm_requests,
                    max_workers=self.max_litellm_workers,
                    **self.litellm_batch_completion_kwargs,
                )
            ]
        else:
            responses = [self.model(messages) for messages in litellm_requests]

        for data, assistant_response in zip(batch, responses, strict=True):
            eval_result = self.evaluator(data, assistant_response)
            score = eval_result.score
            feedback = eval_result.feedback
            obj_scores = eval_result.objective_scores

            output: DefaultRolloutOutput = {"full_assistant_response": assistant_response}

            outputs.append(output)
            scores.append(score)
            objective_scores.append(obj_scores)

            if trajectories is not None:
                trajectories.append(
                    {
                        "data": data,
                        "full_assistant_response": assistant_response,
                        "feedback": feedback,
                    }
                )

        objective_scores_arg: list[dict[str, float]] | None = None
        if objective_scores:
            all_none = all(x is None for x in objective_scores)
            all_not_none = all(x is not None for x in objective_scores)
            if not (all_none or all_not_none):
                raise ValueError("Objective scores must either be all None or all not None.")
            if all_not_none:
                objective_scores_arg = cast(list[dict[str, float]], objective_scores)

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores_arg,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[DefaultTrajectory, DefaultRolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        ret_d: dict[str, list[DefaultReflectiveRecord]] = {}

        assert len(components_to_update) == 1
        comp = components_to_update[0]

        trajectories = eval_batch.trajectories
        assert trajectories is not None, "Trajectories are required to build a reflective dataset."

        items: list[DefaultReflectiveRecord] = []

        for traj in trajectories:
            d: DefaultReflectiveRecord = {
                "Inputs": traj["data"]["input"],
                "Generated Outputs": traj["full_assistant_response"],
                "Feedback": traj["feedback"],
            }
            items.append(d)

        ret_d[comp] = items

        if len(items) == 0:
            raise Exception("No valid predictions found for any module.")

        return ret_d


@app.command()
def main(
    max_metric_calls: int = typer.Option(1, help="Maximum number of metric calls to make."),
    split_counts: int = typer.Option(2, help="Number of splits to make."),
):
    script_path = Path(__file__).name
    logger.info(f"Starting GEPA run {__file__}")
    logger.info(f"{max_metric_calls=}, {split_counts=}")
    assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"
    assert max_metric_calls > 0, "max_metric_calls must be greater than 0"

    trainset, valset, testset = init_dataset_default_adapter(split_counts=split_counts)
    seed_prompt = get_seed_prompt()

    task_lm = "openai/gpt-4.1-mini"
    reflection_lm = "openai/gpt-5.1"

    evaluator: Evaluator | None = GNAFSticklerEvaluator()
    active_adapter: GEPAAdapter | None = CustomGEPAAdapter(model=task_lm, evaluator=evaluator)

    t0 = time.time()
    logger.info(f"Seed system_prompt: {seed_prompt['system_prompt']}")
    logger.info("Running GEPA optimization process...")
    gepa_result = optimize(
        seed_candidate=seed_prompt,
        trainset=trainset,
        valset=valset,
        adapter=active_adapter,
        max_metric_calls=max_metric_calls,
        reflection_lm=reflection_lm,
        track_best_outputs=True,
        display_progress_bar=True,
        logger=log.CustomGepaLogger(),
        use_mlflow=True,
        mlflow_tracking_uri="http://localhost:5001",
        mlflow_experiment_name=script_path,
    )

    log_results(gepa_result, seed_prompt)

    total_seconds = time.time() - t0
    logger.info(f"Done in {total_seconds:.2f} seconds.")

    observability.save_llm_calls(run_id, llm_calls)
    summary_llm_calls = observability.summarize_llm_calls(
        run_id=run_id,
        script_path=script_path,
        llm_calls=llm_calls,
        task_lm=task_lm,
        reflection_lm=reflection_lm,
        total_seconds=total_seconds,
    )
    logger.info(f"LLM calls summary:\n{summary_llm_calls}")

    summary_run = observability.summarize_run(
        run_id,
        script_path,
        seed_prompt,
        trainset,
        valset,
        testset,
        max_metric_calls,
        split_counts,
        task_lm,
        reflection_lm,
        gepa_result,
        summary_llm_calls,
    )

    total_cost_usd = summary_llm_calls["total_cost_usd"]

    logger.info(f"Run summary:\n{summary_run}")
    logger.info(f"Finished GEPA run {__file__}")
    logger.info(f"{total_cost_usd=:.2f}, {total_seconds=:.2f} seconds, {max_metric_calls=}, {split_counts=}")


if __name__ == "__main__":
    log.configure_loguru()
    app()
