import json
import random
from datasets import load_dataset
from loguru import logger
from gepa.adapters.default_adapter.default_adapter import DefaultDataInst


def init_dataset_default_adapter(example_count: int = 100, split_counts: int = 2):
    """
    https://www.kaggle.com/datasets/dylanhogg/geoscape-geocoded-national-address-file-gnaf'

    TODO: custom token instumentation:
    https://chatgpt.com/c/696c8c74-6eac-8323-909e-d2835dc6c80d
    """

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

    logger.info(f"Loaded dataset {len(trainset)=}, {len(valset)=}, {len(testset)=} from {dataset_name=}")
    trainset = trainset[:split_counts]
    valset = valset[:split_counts]
    testset = testset[:split_counts]
    logger.info(f"Dataset trimmed to {len(trainset)=}, {len(valset)=}, {len(testset)=}")

    return trainset, valset, testset


def get_seed_prompt():
    """
    Get the seed prompt for the GNAF dataset.
    """

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

    return seed_prompt


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
