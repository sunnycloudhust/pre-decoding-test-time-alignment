from datasets import load_dataset

dataset = load_dataset("walledai/AdvBench", split="train")

print(dataset)
print(dataset[0])