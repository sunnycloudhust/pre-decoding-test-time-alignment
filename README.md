# Test-time Alignment Experiments

This project compares simple test-time alignment strategies on the AdvBench benchmark. It measures both the AdvBench target-match rate (higher means more unsafe target completion) and a heuristic refusal rate (higher usually means safer behavior).

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

By default, the script compares three inference conditions using the same model:

- `baseline`: the raw benchmark prompt;
- `system`: a safety instruction supplied as a system message;
- `reminder`: a safety reminder appended to the user prompt.

To run a small smoke test:

```bash
python main.py --limit 5 --max-new-tokens 80
```

To run one condition or select a different checkpoint:

```bash
python main.py --methods baseline --model-id Qwen/Qwen2.5-0.5B-Instruct
```

This script will:
- load the model from Hugging Face,
- load the AdvBench dataset,
- generate responses for each prompt,
- compute target-match and refusal rates,
- save one CSV per method and a summary CSV.

## Output

The script generates files such as:

- `qwen_advbench_baseline.csv`
- `qwen_advbench_system.csv`
- `qwen_advbench_reminder.csv`
- `qwen_advbench_summary.csv`

Each row contains:
- `id`
- `prompt`
- `target`
- `response`
- `is_target_match`
- `refusal_detected`

## Notes

The current configuration uses an open-source model:

```python
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
```

If you want to switch to another model, update `MODEL_ID` in `main.py`.

## Interpretation

AdvBench targets are affirmative harmful-completion strings, so `is_target_match=False` does not mean the model failed at safety. For safety comparison, focus on a lower `target_match_rate` and a higher `refusal_rate`. The refusal detector is a simple keyword heuristic and should not replace human or trained safety evaluation.
