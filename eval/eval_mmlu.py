"""
Evaluate model on MMLU subset and save correct/incorrect completions.
Output can be used with generate_vec_accuracy.py to extract steering vectors.
"""
import json
from pathlib import Path
import argparse


def load_mmlu_sample(num_samples=50):
    """Load a small subset of MMLU for quick evaluation."""
    mmlu_samples = [
        {
            "question": "What is the capital of France?",
            "choices": ["London", "Paris", "Berlin", "Madrid"],
            "correct_answer": 1,
        },
        {
            "question": "What is 2 + 2?",
            "choices": ["3", "4", "5", "6"],
            "correct_answer": 1,
        },
    ]
    return mmlu_samples[:num_samples]


def format_mmlu_prompt(sample):
    """Format MMLU question + choices as model input."""
    question = sample["question"]
    choices = sample["choices"]

    prompt = f"Q: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}) {choice}\n"
    prompt += "A: "

    return prompt


def get_correct_completion(sample):
    """Get the correct answer letter."""
    return chr(65 + sample["correct_answer"])


def get_wrong_completions(sample):
    """Get incorrect answer letters."""
    wrong_answers = []
    for i in range(len(sample["choices"])):
        if i != sample["correct_answer"]:
            wrong_answers.append(chr(65 + i))
    return wrong_answers


def evaluate_mmlu(model_name, num_samples=50, save_dir="mmlu_activations"):
    """
    Evaluate model on MMLU and save correct/incorrect completions.

    Returns:
        correct_texts, incorrect_texts, correct_prompts, incorrect_prompts
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading {num_samples} MMLU samples...")
    samples = load_mmlu_sample(num_samples)

    correct_texts = []
    incorrect_texts = []
    correct_prompts = []
    incorrect_prompts = []

    for sample in samples:
        prompt = format_mmlu_prompt(sample)
        correct_ans = get_correct_completion(sample)
        wrong_ans_list = get_wrong_completions(sample)

        correct_texts.append(prompt + correct_ans)
        correct_prompts.append(prompt)

        for wrong_ans in wrong_ans_list:
            incorrect_texts.append(prompt + wrong_ans)
            incorrect_prompts.append(prompt)

    print(f"Collected {len(correct_texts)} correct and {len(incorrect_texts)} incorrect samples")

    metadata = {
        "num_correct": len(correct_texts),
        "num_incorrect": len(incorrect_texts),
        "num_samples": num_samples,
        "model": model_name,
    }

    with open(f"{save_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(f"{save_dir}/correct_completions.json", "w") as f:
        json.dump({"texts": correct_texts, "prompts": correct_prompts}, f)

    with open(f"{save_dir}/incorrect_completions.json", "w") as f:
        json.dump({"texts": incorrect_texts, "prompts": incorrect_prompts}, f)

    print(f"✓ Saved MMLU data to {save_dir}/")
    return correct_texts, incorrect_texts, correct_prompts, incorrect_prompts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on MMLU and save completions for steering vector extraction")
    parser.add_argument("--model_name", type=str, default="/home/yc714/proj/decmas/loopformer/Ouro-1.4B",
                        help="Model name or path")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of MMLU samples to use")
    parser.add_argument("--save_dir", type=str, default="mmlu_activations",
                        help="Directory to save completions")
    args = parser.parse_args()

    evaluate_mmlu(args.model_name, args.num_samples, args.save_dir)
