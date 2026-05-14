import pickle

DATASET_FILES = {
    "hh-rlhf": "initial_red_teaming_data_all.pkl",
    "harmfulqa": "prompt_que_harmfulqa.pkl",
    "dangerousqa": "prompt_que_dangerousqa.pkl",
}

BAD_CASES_FILE = "bad_cases_all_sorted.pkl"


def load_red_teaming_data(dataset_name: str, data_dir: str = "data", max_items: int = 1000) -> list[str]:
    if dataset_name not in DATASET_FILES:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose from: {list(DATASET_FILES)}")

    filepath = f"{data_dir}/{DATASET_FILES[dataset_name]}"
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    if dataset_name == "hh-rlhf":
        data = data[:max_items]

    return data


def load_bad_cases(data_dir: str = "data") -> list:
    filepath = f"{data_dir}/{BAD_CASES_FILE}"
    with open(filepath, "rb") as f:
        return pickle.load(f)
