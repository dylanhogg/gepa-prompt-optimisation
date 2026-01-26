import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, Protocol, TypedDict, cast

import litellm
import typer
from dotenv import load_dotenv
from gepa.api import optimize
from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from loguru import logger

from gnaf_common import get_seed_prompt, init_dataset_default_adapter
from library import log
from library.log import log_results

assert load_dotenv(), "Failed to load .env file"
assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"

app = typer.Typer(pretty_exceptions_enable=False, pretty_exceptions_show_locals=True)

litellm.success_callback = [log.on_litellm_success]
litellm.failure_callback = [log.on_litellm_failure]

"""
Case 4: Custom adapter with structured JSON evaluation (precision/recall/F1)
Uses `StructuredGnafEvaluator` to evaluate the response, and `GnafStructuredAdapter` to wrap the model and evaluator.
(AI produced)
"""


class GnafDataInst(TypedDict):
    input: str
    additional_context: dict[str, str]
    answer: str  # gold JSON string


class EvaluationResult(NamedTuple):
    score: float
    feedback: str
    objective_scores: dict[str, float] | None = None


class ChatMessage(TypedDict):
    role: str
    content: str


class ChatCompletionCallable(Protocol):
    def __call__(self, messages: Sequence[ChatMessage]) -> str: ...


class Evaluator(Protocol):
    def __call__(self, data: GnafDataInst, response: str) -> EvaluationResult: ...


class StructuredGnafEvaluator:
    """
    Evaluates a predicted JSON object against a gold JSON object using:
    - normalized value matching
    - precision/recall/F1 with policy: extra predicted keys => FP, missing gold keys => FN
    - wrong value for a gold key => FP + FN
    """

    _collapse_ws_re = re.compile(r"\s+")

    def __init__(
        self,
        lowercase_fields: set[str] | None = None,
        strip_edge_punctuation: bool = True,
    ):
        self.lowercase_fields = lowercase_fields or {
            "state_abbreviation",
            "street_type_code",
            "street_suffix_code",
            "flat_type",
            "level_type",
        }
        self.strip_edge_punctuation = strip_edge_punctuation

    def __call__(self, data: GnafDataInst, response: str) -> EvaluationResult:
        gold_obj, gold_err = parse_json_object(data["answer"])
        if gold_obj is None:
            raise ValueError(f"Gold answer is not valid JSON object: {gold_err}")

        pred_obj, pred_err = parse_json_object(response)
        if pred_obj is None:
            feedback = "\n".join(
                [
                    "Invalid JSON output (must be a single JSON object).",
                    f"Parse error: {pred_err}",
                    "Fix: output ONLY JSON, no extra text, no code fences.",
                ]
            )
            return EvaluationResult(
                score=0.0,
                feedback=feedback,
                objective_scores={
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "json_valid": 0.0,
                    "extra_keys": 0.0,
                },
            )

        gold_keys = set(gold_obj.keys())
        pred_keys = set(pred_obj.keys())

        tp = 0
        fp = 0
        fn = 0

        extras = sorted(k for k in pred_keys if k not in gold_keys)
        missing = sorted(k for k in gold_keys if k not in pred_keys)
        mismatched: list[tuple[str, str, str]] = []

        # Predicted keys: classify TP/FP (+FN for wrong values on gold keys).
        for k in sorted(pred_keys):
            if k not in gold_keys:
                fp += 1
                continue

            gold_norm = normalize_value(
                field=k,
                value=gold_obj.get(k),
                lowercase_fields=self.lowercase_fields,
                strip_edge_punctuation=self.strip_edge_punctuation,
            )
            pred_norm = normalize_value(
                field=k,
                value=pred_obj.get(k),
                lowercase_fields=self.lowercase_fields,
                strip_edge_punctuation=self.strip_edge_punctuation,
            )

            if pred_norm == gold_norm:
                tp += 1
            else:
                fp += 1
                fn += 1
                mismatched.append((k, gold_norm, pred_norm))

        # Missing gold keys: FN
        fn += len(missing)

        precision = safe_div(tp, tp + fp, default=(1.0 if len(gold_keys) == 0 else 0.0))
        recall = safe_div(tp, tp + fn, default=1.0)
        f1 = safe_f1(precision, recall)

        feedback_lines = [
            "Structured evaluation (normalized field matching).",
            f"TP={tp} FP={fp} FN={fn} precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}",
        ]

        if extras:
            feedback_lines.append(f"Unexpected keys (remove): {', '.join(extras)}")
        if missing:
            feedback_lines.append(f"Missing keys (add): {', '.join(missing)}")
        if mismatched:
            feedback_lines.append("Mismatched values:")
            for k, g, p in mismatched[:20]:
                feedback_lines.append(f"- {k}: expected={json.dumps(g)} got={json.dumps(p)}")
            if len(mismatched) > 20:
                feedback_lines.append(f"- ... and {len(mismatched) - 20} more mismatches")

        return EvaluationResult(
            score=f1,
            feedback="\n".join(feedback_lines),
            objective_scores={
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "json_valid": 1.0,
                "extra_keys": float(len(extras)),
            },
        )


class Trajectory(TypedDict):
    data: GnafDataInst
    full_assistant_response: str
    feedback: str


class RolloutOutput(TypedDict):
    full_assistant_response: str


ReflectiveRecord = TypedDict(
    "ReflectiveRecord",
    {
        "Inputs": str,
        "Generated Outputs": str,
        "Feedback": str,
    },
)


class GnafStructuredAdapter(GEPAAdapter[GnafDataInst, Trajectory, RolloutOutput]):
    def __init__(
        self,
        model: str | ChatCompletionCallable,
        evaluator: Evaluator,
        max_litellm_workers: int = 10,
        litellm_batch_completion_kwargs: dict[str, Any] | None = None,
    ):
        self.model = model
        self.evaluator = evaluator
        self.max_litellm_workers = max_litellm_workers
        self.litellm_batch_completion_kwargs = litellm_batch_completion_kwargs or {}

    def evaluate(
        self,
        batch: list[GnafDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, RolloutOutput]:
        outputs: list[RolloutOutput] = []
        scores: list[float] = []
        objective_scores: list[dict[str, float]] = []
        trajectories: list[Trajectory] | None = [] if capture_traces else None

        system_content = next(iter(candidate.values()))

        requests: list[list[ChatMessage]] = []
        for data in batch:
            requests.append(
                [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": data["input"]},
                ]
            )

        if isinstance(self.model, str):
            responses = [
                (resp.choices[0].message.content or "").strip()
                for resp in litellm.batch_completion(
                    model=self.model,
                    messages=requests,
                    max_workers=self.max_litellm_workers,
                    **self.litellm_batch_completion_kwargs,
                )
            ]
        else:
            responses = [self.model(messages) for messages in requests]

        for data, assistant_response in zip(batch, responses, strict=True):
            eval_result = self.evaluator(data, assistant_response)

            outputs.append({"full_assistant_response": assistant_response})
            scores.append(float(eval_result.score))
            objective_scores.append(eval_result.objective_scores or {})

            if trajectories is not None:
                trajectories.append(
                    {
                        "data": data,
                        "full_assistant_response": assistant_response,
                        "feedback": eval_result.feedback,
                    }
                )

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=cast(list[dict[str, float]], objective_scores),
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Trajectory, RolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        assert len(components_to_update) == 1
        comp = components_to_update[0]

        trajectories = eval_batch.trajectories
        assert trajectories is not None, "Trajectories are required to build a reflective dataset."

        items: list[ReflectiveRecord] = []
        for traj in trajectories:
            items.append(
                {
                    "Inputs": traj["data"]["input"],
                    "Generated Outputs": traj["full_assistant_response"],
                    "Feedback": traj["feedback"],
                }
            )

        if len(items) == 0:
            raise Exception("No valid predictions found for any module.")

        return {comp: items}


def safe_div(num: float, den: float, default: float) -> float:
    if den == 0:
        return float(default)
    return float(num / den)


def safe_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    s = (text or "").strip()
    if not s:
        return None, "Empty response"

    # Attempt 1: direct JSON parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return cast(dict[str, Any], obj), None
        return None, "Parsed JSON is not an object"
    except Exception as e1:
        # Attempt 2: extract first {...} span
        try:
            start = s.find("{")
            end = s.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None, f"Not a JSON object: {e1}"
            candidate = s[start : end + 1]
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return cast(dict[str, Any], obj), None
            return None, "Extracted JSON is not an object"
        except Exception as e2:
            return None, f"{e1}; extracted-span parse failed: {e2}"


def normalize_value(
    *,
    field: str,
    value: Any,
    lowercase_fields: set[str],
    strip_edge_punctuation: bool,
) -> str:
    if value is None:
        s = ""
    elif isinstance(value, (dict, list)):
        s = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        s = str(value)

    s = s.strip()
    s = StructuredGnafEvaluator._collapse_ws_re.sub(" ", s)

    if strip_edge_punctuation:
        s = s.strip(" ,.;:")

    if field in lowercase_fields:
        s = s.lower()

    return s


@app.command()
def main(max_metric_calls: int = typer.Option(10, help="Maximum number of metric calls to make.")):
    logger.info(f"Starting GEPA run {__file__}")
    logger.info(f"{max_metric_calls=}")
    assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"
    assert max_metric_calls > 0, "max_metric_calls must be greater than 0"

    trainset, valset, testset = init_dataset_default_adapter()
    _ = testset

    seed_prompt = get_seed_prompt()

    task_lm = "openai/gpt-4.1-mini"
    evaluator = StructuredGnafEvaluator()
    adapter: GEPAAdapter | None = GnafStructuredAdapter(model=task_lm, evaluator=evaluator)

    t0 = time.time()
    logger.info(f"Seed system_prompt: {seed_prompt['system_prompt']}")
    logger.info("Running GEPA optimization process...")

    gepa_result = optimize(
        seed_candidate=seed_prompt,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        max_metric_calls=max_metric_calls,
        reflection_lm="openai/gpt-5",
        track_best_outputs=True,
        display_progress_bar=True,
        logger=log.CustomGepaLogger(),
        use_mlflow=True,
        mlflow_tracking_uri="http://localhost:5001",
        mlflow_experiment_name="gepa-gnaf-structured",
    )

    log_results(gepa_result, seed_prompt)

    t1 = time.time() - t0
    logger.info(f"Done in {t1:.2f} seconds.")
    logger.info(f"Finished GEPA run {__file__}")


if __name__ == "__main__":
    log.configure_loguru()
    app()
