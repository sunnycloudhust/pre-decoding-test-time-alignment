import json

from datasets import load_dataset

BENCHMARKS = ("advbench", "truthfulqa", "safeedit")
# Names of the benchmark datasets:
"""
    advbench: harmful prompts used to evaluate model safety.
    truthfulqa: questions used to evaluate whether answers are truthful.
    safeedit: prompts from the SafeEdit dataset used to evaluate safe editing.
"""

# This function returns the question/prompt/instruction
def _first_value(sample, keys, default=""):
    for key in keys:
        value = sample.get(key)
        if value not in (None, ""):
            return value
    return default

# This function returns and downloads the benchmark
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

# This function changes the form of each benchmark into a general form (prompt, target, metadata)
def normalize_sample(sample, benchmark):
    if benchmark == "advbench":
        return sample["prompt"], sample.get("target", ""), {}

    if benchmark == "truthfulqa":
        metadata = {
            "correct_answers": sample.get("correct_answers", []),
            "incorrect_answers": sample.get("incorrect_answers", []),
        }
        return sample["question"], sample.get("best_answer", ""), metadata
    
    if benchmark == "safeedit":
        prompt = _first_value(sample, ("prompt", "instruction", "question", "input"))
        target = _first_value(sample, ("target", "answer", "response", "output"))
        if not prompt:
            raise ValueError("SafeEdit record has no prompt/instruction/question/input field")
        return prompt, target, {}


def main():
    # SafeEdit is loaded from this local file because it requires manual access.
    safeedit_file = "SafeEdit_test.json"

    # Load and inspect all three benchmarks in the order defined by BENCHMARKS.
    for benchmark in BENCHMARKS:
        print(f"\n--- {benchmark} ---")
        if benchmark == "safeedit":
            dataset = load_benchmark(benchmark, safeedit_file)
        else:
            dataset = load_benchmark(benchmark)

        print("Number of samples:", len(dataset))
        prompt, target, metadata = normalize_sample(dataset[0], benchmark)
        print("Prompt:", prompt)
        print("Target:", target)
        print("Metadata:", metadata)


if __name__ == "__main__":
    main()

