"""
Extract steering vectors for accuracy (correct vs incorrect answers) on MMLU-like tasks.
Reuses extract_hidden_states from generate_vec.py for common activation extraction.
"""
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import argparse
from pathlib import Path
from generate_vec import extract_hidden_states


def extract_accuracy_vectors(model_name, correct_path, incorrect_path, save_dir, num_ut_steps=None):
    """Extract steering vectors for accuracy (correct - incorrect)."""
    save_dir_path = Path(save_dir)

    # Load model config
    config_path = os.path.join(model_name, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config_ut_steps = config.get("total_ut_steps", 1)

    if num_ut_steps is None:
        num_ut_steps = config_ut_steps

    # Check if vectors already exist
    required_files = []
    for ut_step in range(num_ut_steps):
        required_files.extend([
            f"accuracy_prompt_avg_diff_ut{ut_step}.pt",
            f"accuracy_response_avg_diff_ut{ut_step}.pt",
            f"accuracy_prompt_last_diff_ut{ut_step}.pt",
        ])

    all_exist = all((save_dir_path / f).exists() for f in required_files)
    if all_exist:
        print(f"✓ Accuracy steering vectors already exist. Skipping extraction.")
        return

    print(f"🔄 Extracting accuracy steering vectors ({num_ut_steps} UT steps)...")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load correct/incorrect texts
    with open(correct_path) as f:
        correct_data = json.load(f)
    with open(incorrect_path) as f:
        incorrect_data = json.load(f)

    correct_texts = correct_data["texts"]
    correct_prompts = correct_data["prompts"]
    incorrect_texts = incorrect_data["texts"]
    incorrect_prompts = incorrect_data["prompts"]

    print(f"Extracting activations from {len(correct_texts)} correct and {len(incorrect_texts)} incorrect samples...")

    # Extract activations using shared function
    correct_prompt_avg, correct_prompt_last, correct_response_avg = extract_hidden_states(
        model, tokenizer, correct_texts, correct_prompts
    )
    incorrect_prompt_avg, incorrect_prompt_last, incorrect_response_avg = extract_hidden_states(
        model, tokenizer, incorrect_texts, incorrect_prompts
    )

    num_hidden_layers = model.config.num_hidden_layers

    os.makedirs(save_dir, exist_ok=True)

    # Compute and save differences
    for ut_step in range(num_ut_steps):
        diffs_prompt_avg = []
        diffs_response_avg = []
        diffs_prompt_last = []

        for l in range(num_hidden_layers):
            if isinstance(correct_prompt_avg[ut_step][l], torch.Tensor) and isinstance(incorrect_prompt_avg[ut_step][l], torch.Tensor):
                diffs_prompt_avg.append(correct_prompt_avg[ut_step][l].mean(0).float() - incorrect_prompt_avg[ut_step][l].mean(0).float())
            if isinstance(correct_response_avg[ut_step][l], torch.Tensor) and isinstance(incorrect_response_avg[ut_step][l], torch.Tensor):
                diffs_response_avg.append(correct_response_avg[ut_step][l].mean(0).float() - incorrect_response_avg[ut_step][l].mean(0).float())
            if isinstance(correct_prompt_last[ut_step][l], torch.Tensor) and isinstance(incorrect_prompt_last[ut_step][l], torch.Tensor):
                diffs_prompt_last.append(correct_prompt_last[ut_step][l].mean(0).float() - incorrect_prompt_last[ut_step][l].mean(0).float())

        if diffs_prompt_avg:
            prompt_avg_diff = torch.stack(diffs_prompt_avg, dim=0)
        else:
            print(f"  Warning: No prompt_avg data for UT step {ut_step}, skipping")
            continue

        if diffs_response_avg:
            response_avg_diff = torch.stack(diffs_response_avg, dim=0)
        else:
            print(f"  Warning: No response_avg data for UT step {ut_step}, skipping")
            continue

        if diffs_prompt_last:
            prompt_last_diff = torch.stack(diffs_prompt_last, dim=0)
        else:
            print(f"  Warning: No prompt_last data for UT step {ut_step}, skipping")
            continue

        torch.save(prompt_avg_diff, f"{save_dir}/accuracy_prompt_avg_diff_ut{ut_step}.pt")
        torch.save(response_avg_diff, f"{save_dir}/accuracy_response_avg_diff_ut{ut_step}.pt")
        torch.save(prompt_last_diff, f"{save_dir}/accuracy_prompt_last_diff_ut{ut_step}.pt")

    print(f"✓ Accuracy steering vectors saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract accuracy steering vectors from saved MMLU completions")
    parser.add_argument("--model_name", type=str, default="/home/yc714/proj/decmas/loopformer/Ouro-1.4B",
                        help="Model name or path")
    parser.add_argument("--correct_path", type=str, required=True,
                        help="Path to JSON file with correct completions")
    parser.add_argument("--incorrect_path", type=str, required=True,
                        help="Path to JSON file with incorrect completions")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save steering vectors")
    parser.add_argument("--num_ut_steps", type=int, default=None,
                        help="Override number of UT steps (for test-time scaling)")
    args = parser.parse_args()

    extract_accuracy_vectors(args.model_name, args.correct_path, args.incorrect_path, args.save_dir, args.num_ut_steps)
