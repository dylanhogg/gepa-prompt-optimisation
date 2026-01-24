import os
import time
from gepa.adapters.default_adapter.default_adapter import DefaultAdapter, Evaluator
from gepa.core.adapter import GEPAAdapter
import typer
from loguru import logger
from dotenv import load_dotenv
import litellm
from gepa.api import optimize

from gnaf_common import get_seed_prompt, init_dataset_default_adapter
from library.log import log_results
from library import log

assert load_dotenv(), "Failed to load .env file"
assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"

app = typer.Typer(pretty_exceptions_enable=False, pretty_exceptions_show_locals=True)

litellm.success_callback = [log.on_litellm_success]
litellm.failure_callback = [log.on_litellm_failure]

"""
Case 2: Explicitly use builtin default GEPA adapter
Explicitly sets `gepa.api.optimize` to use the builtin default GEPA adapter `gepa.adapters.default_adapter.DefaultAdapter`
"""


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
    evaluator: Evaluator | None = None
    active_adapter: GEPAAdapter | None = DefaultAdapter(model=task_lm, evaluator=evaluator)

    # Run GEPA optimization process.
    t0 = time.time()
    logger.info(f"Seed system_prompt: {seed_prompt['system_prompt']}")
    logger.info("Running GEPA optimization process...")
    gepa_result = optimize(
        seed_candidate=seed_prompt,
        trainset=trainset,
        valset=valset,
        adapter=active_adapter,  # Supply either `adapter` or `task_lm`, but not both
        # task_lm="openai/gpt-4.1-mini",  # <-- This is the model being optimized (only used if `adapter` is not provided)
        max_metric_calls=max_metric_calls,  # <-- Set a budget
        reflection_lm="openai/gpt-5",  # <-- Use a strong model to reflect on mistakes and propose better prompts
        track_best_outputs=True,
        display_progress_bar=True,
        logger=log.CustomGepaLogger(),
        use_mlflow=True,  # Ref: https://dspy.ai/tutorials/gepa_facilitysupportanalyzer/
        mlflow_tracking_uri="http://localhost:5001",
        mlflow_experiment_name="gepa-simple1",
    )

    log_results(gepa_result, seed_prompt)

    t1 = time.time() - t0
    logger.info(f"Done in {t1:.2f} seconds.")


if __name__ == "__main__":
    log.configure_loguru()
    app()
