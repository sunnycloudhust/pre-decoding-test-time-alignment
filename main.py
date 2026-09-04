import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.benchmarks import BENCHMARKS, load_benchmark
from generation import generate_responses
from metrics import compute_accuracy, evaluate_result

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
METHODS = ("baseline", "system", "reminder")
LIMIT = None            # full dataset run
MAX_NEW_TOKENS = 200
BATCH_SIZE = 16
OUTPUT_PREFIX = None


def main():
    if LIMIT is not None and LIMIT < 1:
        raise ValueError("--limit must be at least 1")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype="auto"
    )
    model.eval()

    benchmarks = BENCHMARKS
    for benchmark in benchmarks:
        dataset = load_benchmark(benchmark)
        
        if hasattr(dataset, "keys"):
            split = "test" if "test" in dataset else next(iter(dataset))
            dataset = dataset[split]
        if LIMIT is not None:
            dataset = dataset.select(range(min(LIMIT, len(dataset))))
            
        print(f"\nBenchmark: {benchmark}")
        print("Number of samples:", len(dataset))

        summary = []
        for method in METHODS:
            results = generate_responses(
                dataset,
                model,
                tokenizer,
                benchmark=benchmark,
                method=method,
                max_new_tokens=MAX_NEW_TOKENS,
                batch_size=BATCH_SIZE,
            )
            for result in results:
                metadata = result.pop("metadata", {})
                result.update(
                    evaluate_result(
                        benchmark,
                        result["response"],
                        result["target"],
                        metadata,
                    )
                )
            output_prefix = OUTPUT_PREFIX or f"qwen_{benchmark}"
            if OUTPUT_PREFIX and len(benchmarks) > 1:
                output_prefix = f"{OUTPUT_PREFIX}_{benchmark}"
            pd.DataFrame(results).to_csv(
                f"{output_prefix}_{method}.csv", index=False
            )

            metrics = {
                "method": method,
                "target_match_rate": compute_accuracy(results),
                "refusal_rate": sum(item["refusal_detected"] for item in results)
                / len(results),
                "samples": len(results),
            }
            if benchmark == "truthfulqa":
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

        output_prefix = OUTPUT_PREFIX or f"qwen_{benchmark}"
        if OUTPUT_PREFIX and len(benchmarks) > 1:
            output_prefix = f"{OUTPUT_PREFIX}_{benchmark}"
        pd.DataFrame(summary).to_csv(f"{output_prefix}_summary.csv", index=False)


if __name__ == "__main__":
    main()
