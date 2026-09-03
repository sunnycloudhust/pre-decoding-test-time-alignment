import json

from datasets import load_dataset

BENCHMARKS = ("advbench", "truthfulqa", "safeedit")


def _first_value(sample, keys, default=""):
    for key in keys:
        value = sample.get(key)
        if value not in (None, ""):
            return value
    return default


def load_benchmark(name, safeedit_file=None):
    if name == "advbench":
        return load_dataset("walledai/AdvBench", split="train")

    if name == "truthfulqa":
        return load_dataset("truthfulqa/truthful_qa", "generation", split="validation")

    if name == "safeedit":
        if not safeedit_file:
            raise ValueError(
                "SafeEdit requires --safeedit-file after downloading SafeEdit_test.json "
                "from zjunlp/SafeEdit"
            )
        with open(safeedit_file, encoding="utf-8") as file:
            records = json.load(file)
        if isinstance(records, dict):
            records = records.get("data", records.get("test", []))
        return records

    raise ValueError(f"Unknown benchmark: {name}")


def normalize_sample(sample, benchmark):
    if benchmark == "advbench":
        return sample["prompt"], sample.get("target", ""), {}

    if benchmark == "truthfulqa":
        metadata = {
            "correct_answers": sample.get("correct_answers", []),
            "incorrect_answers": sample.get("incorrect_answers", []),
        }
        return sample["question"], sample.get("best_answer", ""), metadata

    prompt = _first_value(sample, ("prompt", "instruction", "question", "input"))
    target = _first_value(sample, ("target", "answer", "response", "output"))
    if not prompt:
        raise ValueError("SafeEdit record has no prompt/instruction/question/input field")
    return prompt, target, {}
