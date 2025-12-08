import json
from pathlib import Path

RESULTS_DIR = Path("experiments/results")
HPARAM_PATH = RESULTS_DIR / "hparam_results.json"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def log_hparam_result(experiment_id: str, result_dict: dict):
    if HPARAM_PATH.exists():
        with HPARAM_PATH.open() as f:
            data = json.load(f)
    else:
        data = {}

    data[experiment_id] = result_dict

    with HPARAM_PATH.open("w") as f:
        json.dump(data, f, indent=2)
