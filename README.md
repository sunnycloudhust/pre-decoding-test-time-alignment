# Test-time Alignment Experiments

This project evaluates an open-source causal language model on the AdvBench benchmark and records generated responses with an accuracy metric.

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

This script will:
- load the model from Hugging Face,
- load the AdvBench dataset,
- generate responses for each prompt,
- compute a simple target-match accuracy,
- save the results to a CSV file.

## Output

The script generates a CSV file such as:

- `qwen_advbench_results.csv`

Each row contains:
- `id`
- `prompt`
- `target`
- `response`
- `is_target_match`

## Notes

The current configuration uses an open-source model:

```python
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
```

If you want to switch to another model, update `MODEL_ID` in `main.py`.
