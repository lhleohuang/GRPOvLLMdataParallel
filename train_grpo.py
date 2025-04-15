from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from efficient_reasoning.utils import evaluate
from datasets import Dataset
from datetime import datetime
import torch
import argparse

data = []
with open("../data/MATH-500/train.jsonl") as f:
    for line in f:
        tmp = eval(line)
        data.append(tmp)

new_data = []
for index, dict in enumerate(data):
    new_dict = {"prompt": dict["problem"], "answer": dict["answer"]}
    new_data.append(new_dict)

print(f"Length of training set: {len(new_data)}")

dataset = Dataset.from_list(new_data)

def reward(prompts, completions, **kwargs):
    answers = []
    for i, prompt in enumerate(prompts):
        found = False
        for j, example in enumerate(dataset):
            if prompt == example["prompt"]:
                answers.append(example["answer"])
                found = True
                break
        if not found:
            raise ValueError(f"Cannot find problem in dataset: {prompt}")

    return evaluate("MATH-500", completions, answers)

server_configs = [
    {"host": "158.130.55.13", "server_port": 8000, "group_port": 51216},
    {"host": "158.130.55.13", "server_port": 8010, "group_port": 51217},
]

training_args = GRPOConfig(
    output_dir=f"/data0/leoh/grpo-qwen2.5-0.5b-base", 
    logging_steps=100, 
    per_device_train_batch_size=2,
    use_vllm=True, 
    num_generations=8, 
    scale_rewards=False, 
    save_steps=200, 
    save_strategy="steps",
    max_completion_length=2048,
    vllm_server_configs=server_configs,
    )
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B",
    reward_funcs=reward,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
