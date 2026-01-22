import json
import os
import time

from gnaf_custom_adapter_classes import DefaultAdapter, DefaultDataInst, Evaluator
from gepa.core.adapter import GEPAAdapter
import typer
from loguru import logger
from dotenv import load_dotenv
import litellm
from gepa.api import optimize

from library import log

assert load_dotenv(), "Failed to load .env file"
assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"

app = typer.Typer(pretty_exceptions_enable=False, pretty_exceptions_show_locals=True)

litellm.success_callback = [log.on_litellm_success]
litellm.failure_callback = [log.on_litellm_failure]


def init_dataset(example_count: int = 100):
    """
    https://www.kaggle.com/datasets/dylanhogg/geoscape-geocoded-national-address-file-gnaf'

    TODO: custom token instumentation:
    https://chatgpt.com/c/696c8c74-6eac-8323-909e-d2835dc6c80d
    """

    import random

    from datasets import load_dataset

    dataset_name = "dylanhogg/gnaf-2022-structured-training-100000-v0-instruct"

    train_split: list[DefaultDataInst] = [
        {
            # NOTE: GEPA's DefaultAdapter is typed to DefaultDataInst; force concrete `str` types
            # so Pyright doesn't infer dict[str, Unknown] from HuggingFace dataset `.get(...)`.
            "input": str(x.get("input") or ""),  # pyright: ignore[reportAttributeAccessIssue]
            "additional_context": {"solution": str(x.get("output") or "")},  # pyright: ignore[reportAttributeAccessIssue]
            "answer": str(x.get("output") or ""),  # pyright: ignore[reportAttributeAccessIssue]
        }
        for x in load_dataset(dataset_name)["train"]
    ]
    train_split = train_split[:example_count]
    random.Random(0).shuffle(train_split)
    # test_split = [
    #     {"input": x["problem"], "answer": "### " + str(x["answer"])}
    #     for x in load_dataset("MathArena/aime_2025")["train"]
    # ]

    trainset = train_split[: len(train_split) // 2]
    valset = train_split[len(train_split) // 2 :]
    # testset = test_split * 5
    testset = []  # TODO

    logger.info(f"Loaded {len(trainset)=}, {len(valset)=}, {len(testset)=} from {dataset_name=}")

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

    trainset, valset, testset = init_dataset()
    logger.info(f"Loaded {len(trainset)=}, {len(valset)=}")
    trainset = trainset[:4]
    valset = valset[:2]
    logger.info(f"Trimmed to {len(trainset)=}, {len(valset)=}")

    gnaf_fields = [
        "building_name",
        "flat_number",
        "flat_number_prefix",
        "flat_number_suffix",
        "flat_type",
        # "latitude",
        "level_number",
        "level_number_prefix",
        "level_number_suffix",
        "level_type",
        "locality_name",
        # "longitude",
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

    gnaf_fields_str = ", ".join(gnaf_fields)

    seed_prompt = {
        "system_prompt": f"You are a helpful assistant. You are given an Australian text address. "
        f"The answer should be structured json with the Geoscape Geocoded National Address File (GNAF) key fields: `{gnaf_fields_str}`. "
    }

    task_lm = "openai/gpt-4.1-mini"  # <-- This is the model being optimized

    # evaluator: Evaluator | None = None,
    # evaluator = None
    # active_adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput] | None = None
    # active_adapter = cast(
    #     GEPAAdapter[DataInst, Trajectory, RolloutOutput], DefaultAdapter(model=task_lm, evaluator=evaluator)
    # )

    evaluator: Evaluator | None = None
    active_adapter: GEPAAdapter | None = DefaultAdapter(model=task_lm, evaluator=evaluator)

    # Let's run GEPA optimization process.
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

    t1 = time.time() - t0
    logger.info(f"Done in {t1:.2f} seconds.")


if __name__ == "__main__":
    log.configure_loguru()
    app()
