import os
import random
import time
from pathlib import Path

import gepa
import litellm
import typer
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from loguru import logger

from library import llm, log

assert load_dotenv(), "Failed to load .env file"
assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"

app = typer.Typer(pretty_exceptions_enable=False, pretty_exceptions_show_locals=True)


litellm.success_callback = [llm.on_litellm_success]
litellm.failure_callback = [llm.on_litellm_failure]
# litellm._turn_on_debug()


def init_dataset():
    """
    # Datasets for gepa.examples.aime:
    # https://github.com/gepa-ai/gepa/blob/v0.0.24/src/gepa/examples/aime.py
    # trainset & valset: https://huggingface.co/datasets/AI-MO/aimo-validation-aime
    # testset (unused here): https://huggingface.co/datasets/MathArena/aime_2025
    """

    train_ds: Dataset = load_dataset("AI-MO/aimo-validation-aime")["train"]
    train_split = [
        {
            "input": x.get("problem"),  # pyright: ignore[reportAttributeAccessIssue]
            "additional_context": {"solution": x.get("solution")},  # pyright: ignore[reportAttributeAccessIssue]
            "answer": "### " + str(x.get("answer")),  # pyright: ignore[reportAttributeAccessIssue]
        }
        for x in train_ds
    ]
    random.Random(0).shuffle(train_split)
    test_ds: Dataset = load_dataset("MathArena/aime_2025")["train"]
    test_split = [
        {"input": x.get("problem"), "answer": "### " + str(x.get("answer"))}  # pyright: ignore[reportAttributeAccessIssue]
        for x in test_ds
    ]

    trainset = train_split[: len(train_split) // 2]
    valset = train_split[len(train_split) // 2 :]
    testset = test_split * 5

    return trainset, valset, testset


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

    # Load AIME dataset
    logger.info("Loading dataset...")

    # Datasets for gepa.examples.aime:
    # https://github.com/gepa-ai/gepa/blob/v0.0.24/src/gepa/examples/aime.py
    # trainset & valset: https://huggingface.co/datasets/AI-MO/aimo-validation-aime
    # testset (unused here): https://huggingface.co/datasets/MathArena/aime_2025
    # trainset, valset, testset = gepa.examples.aime.init_dataset()
    trainset, valset, testset = init_dataset()
    logger.info(f"Loaded {len(trainset)=}, {len(valset)=}")
    trainset = trainset[:4]
    valset = valset[:2]
    logger.info(f"Trimmed to {len(trainset)=}, {len(valset)=}")

    seed_prompt = {
        "system_prompt": "You are a helpful assistant. You are given a question and you need to answer it. "
        "The answer should be given at the end of your response in exactly the format '### <final answer>'"
    }

    # Let's run GEPA optimization process.
    t0 = time.time()
    logger.info("Running GEPA optimization process...")
    gepa_result = gepa.api.optimize(
        seed_candidate=seed_prompt,
        trainset=trainset,
        valset=valset,
        task_lm="openai/gpt-4.1-mini",  # <-- This is the model being optimized
        max_metric_calls=max_metric_calls,  # <-- Set a budget
        reflection_lm="openai/gpt-5",  # <-- Use a strong model to reflect on mistakes and propose better prompts
        track_best_outputs=True,
        display_progress_bar=True,
        logger=log.CustomGepaLogger(),
        use_mlflow=True,  # Ref: https://dspy.ai/tutorials/gepa_facilitysupportanalyzer/
        mlflow_tracking_uri="http://localhost:5001",
        mlflow_experiment_name=Path(__file__).name,
    )

    logger.info("GEPA Optimized Prompt:", gepa_result.best_candidate["system_prompt"])
    logger.info(f"{gepa_result=}")

    t1 = time.time() - t0
    logger.info(f"Done in {t1:.2f} seconds.")


if __name__ == "__main__":
    log.configure_loguru()
    app()
