from stickler.structured_object_evaluator.models.structured_model import StructuredModel


def test_stickler_eval_structured_model() -> None:
    # Define model configuration
    config = {
        "model_name": "Product",
        "match_threshold": 0.8,
        "fields": {
            "name": {"type": "str", "comparator": "LevenshteinComparator", "threshold": 0.8, "weight": 2.0},
            "price": {"type": "float", "comparator": "NumericComparator", "default": 0.0},
        },
    }

    # Create dynamic model class
    Product = StructuredModel.model_from_json(config)

    # Use like any Pydantic model
    product1 = Product(name="Widget", price=29.99)  # pyright: ignore[reportCallIssue]
    product2 = Product(name="Gadget", price=29.99)  # pyright: ignore[reportCallIssue]

    # Full comparison capabilities
    result = product1.compare_with(product2)
    similarity = result["overall_score"]
    print(f"Similarity: {similarity:.2f}")
    assert (0.33 - similarity) < 1e-6


def test_stickler_eval_json() -> None:
    # Define model configuration
    config = {
        "model_name": "Product",
        "match_threshold": 0.8,
        "fields": {
            "name": {"type": "str", "comparator": "LevenshteinComparator", "threshold": 0.8, "weight": 2.0},
            "price": {"type": "float", "comparator": "NumericComparator", "default": 0.0},
        },
    }

    # Create dynamic model class
    Product = StructuredModel.model_from_json(config)

    ground_truth_dict = {"name": "Widget", "price": 29.99}
    ground_truth = Product(**ground_truth_dict)

    prediction_dict = {"name": "Gadget", "price": 29.99}
    prediction = Product(**prediction_dict)

    # Full comparison capabilities
    result = ground_truth.compare_with(prediction)
    similarity = result["overall_score"]
    print(f"Similarity: {similarity:.2f}")
    assert (0.33 - similarity) < 1e-6
