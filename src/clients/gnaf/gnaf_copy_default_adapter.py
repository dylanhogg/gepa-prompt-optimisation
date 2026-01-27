import os
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
from gnaf_common import get_seed_prompt, init_dataset_default_adapter, log_results
from loguru import logger

from library import llm, log, observability

assert load_dotenv(), "Failed to load .env file"
assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"

app = typer.Typer(pretty_exceptions_enable=False, pretty_exceptions_show_locals=True)
run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"

llm_calls = []
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


# TODO: should we be using async callbacks? https://docs.litellm.ai/docs/observability/custom_callback#async-callback-functions
litellm.input_callback = [on_litellm_input]
litellm.success_callback = [on_litellm_success]
litellm.failure_callback = [on_litellm_failure]


"""
Case 3: Copy of default GEPA adapter classes
Copies the default GEPA adapter classes from `gepa.adapters.default_adapter.DefaultAdapter` (v0.0.25) with no real modifications.
"""


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


# Callable that evaluates a response and returns (score, feedback, optional objective_scores)
class Evaluator(Protocol):
    def __call__(self, data: DefaultDataInst, response: str) -> EvaluationResult:
        """
        Evaluates a response and returns a score, feedback, and optional objective scores.
        """
        ...


class ContainsAnswerEvaluator:
    """Default evaluator that checks if the expected answer is contained in the response."""

    def __init__(self, failure_score: float = 0.0):
        self.failure_score = failure_score

    def __call__(self, data: DefaultDataInst, response: str) -> EvaluationResult:
        is_correct = data["answer"] in response
        score = 1.0 if is_correct else self.failure_score

        if is_correct:
            feedback = f"The generated response is correct. The response include the correct answer '{data['answer']}'"
        else:
            additional_context_str = "\n".join(f"{k}: {v}" for k, v in data["additional_context"].items())
            feedback = (
                f"The generated response is incorrect. The correct answer is '{data['answer']}'. "
                "Ensure that the correct answer is included in the response exactly as it is."
            )
            if additional_context_str:
                feedback += f" Here is some additional context that might be helpful:\n{additional_context_str}"

        return EvaluationResult(score=score, feedback=feedback, objective_scores=None)


class CustomGEPAAdapter(GEPAAdapter[DefaultDataInst, DefaultTrajectory, DefaultRolloutOutput]):
    def __init__(
        self,
        model: str | ChatCompletionCallable,
        evaluator: Evaluator | None = None,  # TODO: make required
        max_litellm_workers: int = 10,
        litellm_batch_completion_kwargs: dict[str, Any] | None = None,
    ):
        if isinstance(model, str):
            import litellm

            self.litellm = litellm
        self.model = model
        self.evaluator = evaluator or ContainsAnswerEvaluator()
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
def main(max_metric_calls: int = typer.Option(1, help="Maximum number of metric calls to make.")):
    """
    The core of this example has been adapted from the homepage example using gepa.examples.aime.init_dataset():
    https://github.com/gepa-ai/gepa/tree/v0.0.24?tab=readme-ov-file#simple-prompt-optimization-example
    """

    logger.info(f"Starting GEPA run {__file__}")
    logger.info(f"{max_metric_calls=}")
    assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"
    assert max_metric_calls > 0, "max_metric_calls must be greater than 0"

    trainset, valset, testset = init_dataset_default_adapter()
    seed_prompt = get_seed_prompt()

    task_lm = "openai/gpt-4.1-mini"  # <-- This is the model being optimized
    reflection_lm = "openai/gpt-5"  # <-- Use a strong model to reflect on mistakes and propose better prompts

    # evaluator: Evaluator | None = None,
    # evaluator = None
    # active_adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput] | None = None
    # active_adapter = cast(
    #     GEPAAdapter[DataInst, Trajectory, RolloutOutput], DefaultAdapter(model=task_lm, evaluator=evaluator)
    # )

    evaluator: Evaluator | None = None
    active_adapter: GEPAAdapter | None = CustomGEPAAdapter(model=task_lm, evaluator=evaluator)

    # Run GEPA optimization process.
    t0 = time.time()
    logger.info(f"Seed system_prompt: {seed_prompt['system_prompt']}")
    logger.info("Running GEPA optimization process...")
    gepa_result = optimize(
        seed_candidate=seed_prompt,
        trainset=trainset,
        valset=valset,
        adapter=active_adapter,  # Supply either `adapter` or `task_lm`, but not both
        max_metric_calls=max_metric_calls,  # <-- Set a budget
        reflection_lm=reflection_lm,
        track_best_outputs=True,
        display_progress_bar=True,
        logger=log.CustomGepaLogger(),
        use_mlflow=True,  # Ref: https://dspy.ai/tutorials/gepa_facilitysupportanalyzer/
        mlflow_tracking_uri="http://localhost:5001",
        mlflow_experiment_name=Path(__file__).name,
    )

    log_results(gepa_result, seed_prompt)

    t1 = time.time() - t0
    logger.info(f"Done in {t1:.2f} seconds.")

    observability.save_llm_calls(run_id, llm_calls)
    summary_llm_calls = observability.summarize_llm_calls(
        run_id, Path(__file__).name, llm_calls, task_lm, reflection_lm
    )
    logger.info(f"LLM calls summary:\n{summary_llm_calls}")

    summary_run = observability.summarize_run(
        run_id,
        Path(__file__).name,
        seed_prompt,
        trainset,
        valset,
        testset,
        max_metric_calls,
        task_lm,
        reflection_lm,
        gepa_result,
        summary_llm_calls,
    )
    logger.info(f"Run summary:\n{summary_run}")


if __name__ == "__main__":
    log.configure_loguru()
    app()
