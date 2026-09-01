import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# 1. Load model
# =========================

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto"
)

model.eval()


# =========================
# 2. Load benchmark
# =========================

dataset = load_dataset(
    "walledai/AdvBench",
    split="train"
)

print("Number of samples:", len(dataset))


# =========================
# 3. Generate
# =========================

results = []

for i, sample in enumerate(tqdm(dataset)):

    prompt = sample["prompt"]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Chỉ lấy phần model generate,
    # bỏ prompt ban đầu
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    results.append({
        "id": i,
        "prompt": prompt,
        "response": response
    })


# =========================
# 4. Save
# =========================

df = pd.DataFrame(results)

df.to_csv(
    "tinyllama_advbench_results.csv",
    index=False
)

print(df.head())