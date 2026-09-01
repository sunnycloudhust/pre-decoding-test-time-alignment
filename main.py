import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_PATH = "qwen_advbench_results.csv"


def compute_accuracy(results):
    """Return the fraction of samples where the expected target appears in the generated response."""
    if not results:
        return 0.0

    correct = sum(1 for item in results if item["is_target_match"])
    return correct / len(results)


def generate_responses(dataset, model, tokenizer, max_new_tokens=200, temperature=0.7, do_sample=True):
    results = []

    for i, sample in enumerate(tqdm(dataset, desc="Generating responses")):
        prompt = sample["prompt"]
        target = sample.get("target", "")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        is_target_match = False
        if target:
            is_target_match = target.lower() in response.lower()

        results.append(
            {
                "id": i,
                "prompt": prompt,
                "target": target,
                "response": response,
                "is_target_match": is_target_match,
            }
        )

    return results


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()

    dataset = load_dataset("walledai/AdvBench", split="train")
    print("Number of samples:", len(dataset))

    results = generate_responses(dataset, model, tokenizer)
    accuracy = compute_accuracy(results)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Accuracy: {accuracy:.4f} ({sum(1 for item in results if item['is_target_match'])}/{len(results)})")
    print(df.head())


if __name__ == "__main__":
    main()