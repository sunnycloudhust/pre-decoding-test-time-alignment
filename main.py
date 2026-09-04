import argparse

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.benchmarks import BENCHMARKS, load_benchmark
from generation import generate_responses
from metrics import compute_accuracy

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_METHODS = ("baseline", "system", "reminder")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate test-time alignment methods on safety and truthfulness benchmarks."
    )
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="advbench")
    parser.add_argument("--safeedit-file", default=None)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--methods", default=','.join(DEFAULT_METHODS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    invalid_methods = set(methods) - set(DEFAULT_METHODS)
    if invalid_methods:
        raise ValueError(f"Unknown methods: {', '.join(sorted(invalid_methods))}")
    if args.output_prefix is None:
        args.output_prefix = f"qwen_{args.benchmark}"

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", torch_dtype="auto"
    )
    model.eval()

    dataset = load_benchmark(args.benchmark, args.safeedit_file)
    if isinstance(dataset, list):
        dataset = dataset[:args.limit] if args.limit is not None else dataset
    elif args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    print("Number of samples:", len(dataset))

    summary = []
    for method in methods:
        results = generate_responses(
            dataset,
            model,
            tokenizer,
            benchmark=args.benchmark,
            method=method,
            max_new_tokens=args.max_new_tokens,
        )
        pd.DataFrame(results).to_csv(
            f"{args.output_prefix}_{method}.csv", index=False
        )

        metrics = {
            "method": method,
            "target_match_rate": compute_accuracy(results),
            "refusal_rate": sum(item["refusal_detected"] for item in results)
            / len(results),
            "samples": len(results),
        }
        if args.benchmark == "truthfulqa":
            metrics["truthful_rate"] = sum(
                item["truthful_response"] for item in results
            ) / len(results)
            metrics["incorrect_rate"] = sum(
                item["incorrect_match"] for item in results
            ) / len(results)
        summary.append(metrics)
        print(
            f"{method}: target_match_rate={metrics['target_match_rate']:.4f}, "
            f"refusal_rate={metrics['refusal_rate']:.4f}"
        )

    pd.DataFrame(summary).to_csv(f"{args.output_prefix}_summary.csv", index=False)


if __name__ == "__main__":
    main()
