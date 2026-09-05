# Pre-decoding test-time alignment

Comparison of simple test-time alignment strategies on AdvBench, TruthfulQA, and SafeEdit. The experiments measure how much lightweight prompting changes a model's behavior at inference time.

## Results

The full-dataset run used `Qwen/Qwen2.5-0.5B-Instruct` and compared the refusal rates of the three inference conditions below. Rates are reported as proportions from 0 to 1.

| Benchmark | Samples | baseline | system | reminder |
| --- | ---: | ---: | ---: | ---: |
| AdvBench | 520 | 0.6038 | 0.9212 | 0.7692 |
| TruthfulQA | 817 | 0.0832 | 0.1542 | 0.1983 |
| SafeEdit | 1,890 | 0.1989 | 0.7735 | 0.5540 |

The table reports `refusal_rate`; higher values indicate stronger refusal behavior. TruthfulQA also has `truthful_rate` and `incorrect_rate` in its summary CSV; these values are not printed by the current console summary.

## Repository Guide

- `main.py` loads the model and benchmarks, runs every configured method, and writes CSV results.
- `generation.py` formats prompts and generates model responses for each experiment condition.
- `metrics.py` computes target-match, refusal, truthfulness, and incorrect-answer indicators.
- `benchmarks/benchmarks.py` loads benchmark data and normalizes records into a shared format.
- `benchmarks/benchmarks.py` also contains the benchmark-specific input handling for AdvBench, TruthfulQA, and SafeEdit.

## Experiment Conditions

- **Baseline**: sends the original benchmark prompt unchanged.
- **System instruction**: adds a safety instruction as a system message when the tokenizer supports chat templates.
- **Reminder**: appends a safety reminder to the user prompt.

The comparison is intentionally small and reproducible: it isolates prompt-level test-time changes without fine-tuning the model or changing the evaluation data.

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

By default, the script evaluates AdvBench and compares three inference conditions using the same model:

- `baseline`: the raw benchmark prompt;
- `system`: a safety instruction supplied as a system message;
- `reminder`: a safety reminder appended to the user prompt.

To run a small smoke test:

```bash
python main.py --limit 5 --max-new-tokens 80
```

Run TruthfulQA:

```bash
python main.py --benchmark truthfulqa --limit 5 --max-new-tokens 120
```

Run SafeEdit after requesting access to `zjunlp/SafeEdit` on Hugging Face and downloading `SafeEdit_test.json`:

```bash
python main.py --benchmark safeedit --safeedit-file /path/to/SafeEdit_test.json \
	--limit 5 --max-new-tokens 120
```

To run one condition or select a different checkpoint:

```bash
python main.py --methods baseline --model-id Qwen/Qwen2.5-0.5B-Instruct
```

The script will load the selected benchmark, generate responses for each prompt, compute benchmark-specific metrics, and save one CSV per method plus a summary CSV. Use `--output-prefix` when running multiple models so their output files do not overwrite one another.

## Output

The script generates files such as:

- `qwen_advbench_baseline.csv`
- `qwen_advbench_system.csv`
- `qwen_advbench_reminder.csv`
- `qwen_advbench_summary.csv`

Each row contains common fields:
- `id`
- `prompt`
- `target`
- `response`
- `is_target_match`
- `refusal_detected`

TruthfulQA additionally records `truthful_match`, `incorrect_match`, and `truthful_response`. Its `truthful_rate` is a lightweight substring-based check; use the official TruthfulQA evaluator or `lm-eval` for paper-quality results.

For SafeEdit, the JSON record must contain one of `prompt`, `instruction`, `question`, or `input`. The loader accepts `target`, `answer`, `response`, or `output` when available. SafeEdit data is access-restricted, so the file must be obtained through the dataset's official access process.

## Notes

The current configuration uses an open-source model:

```python
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
```

If you want to switch to another model, update `MODEL_ID` in `main.py`.

## Interpretation

AdvBench targets are affirmative harmful-completion strings, so `is_target_match=False` does not mean the model failed at safety. For safety comparison, focus on a lower `target_match_rate` and a higher `refusal_rate`. For TruthfulQA, focus on a higher `truthful_rate` and lower `incorrect_rate`. All refusal and substring detectors are lightweight heuristics and should not replace official, human, or trained safety evaluation.
