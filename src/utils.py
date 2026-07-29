import json
import os
from datetime import datetime


def print_banner(title: str, width: int = 65) -> None:
    """Print a formatted section header for pipeline output."""
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_section(title: str) -> None:
    """Print a lighter sub-section header."""
    print(f"\n── {title} {'─' * max(0, 55 - len(title))}")


def save_metrics(
    results: dict,
    t_train: float,
    t_infer: float,
    config: dict,
    output_path: str = None,
) -> None:
    """
    Persist evaluation metrics to a JSON file.

    Skips non-serialisable keys (report string, confusion matrix array)
    so the output file is clean and machine-readable.

    Parameters
    ----------
    results     : dict    — output of evaluate_model()
    t_train     : float   — training time in seconds
    t_infer     : float   — inference time in ms per prediction
    config      : dict    — full config dict (model name, seed, etc.)
    output_path : str, optional — overrides config['output']['metrics_path']
    """
    path = output_path or config.get("output", {}).get(
        "metrics_path", "outputs/metrics.json"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)

    output = {
        "timestamp":          datetime.now().isoformat(),
        "model":              config.get("model", {}).get("name", "Unknown"),
        "random_seed":        config.get("data", {}).get("random_seed", 42),
        "accuracy":           results.get("accuracy"),
        "macro_precision":    results.get("precision"),
        "macro_recall":       results.get("recall"),
        "macro_f1":           results.get("f1"),
        "train_time_s":       t_train,
        "infer_time_ms_pred": t_infer,
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path : str — path to config.yaml

    Returns
    -------
    dict
    """
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def check_importable() -> None:
    """
    Smoke-test that all src modules import cleanly.
    Run via: python -c "from src.utils import check_importable; check_importable()"
    """
    import src.data
    import src.features
    import src.model
    import src.utils
    print("All src modules imported successfully.")
