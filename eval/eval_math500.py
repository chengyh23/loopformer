"""
Evaluate models on MATH-500 (Mathematical reasoning benchmark).
MATH-500: https://huggingface.co/datasets/HuggingFaceH4/MATH-500
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from eval.model_utils import load_model_loopformer, load_model_ouro


def extract_answer(text: str) -> str:
    """
    Extract the final numerical answer from generated text.
    Looks for patterns like "Answer: ...", "answer is ...", "\\boxed{...}", etc.
    """
    # Try boxed format first
    if "\\boxed{" in text:
        try:
            start = text.index("\\boxed{") + len("\\boxed{")
            end = text.index("}", start)
            return text[start:end].strip()
        except (ValueError, IndexError):
            pass

    # Try "Answer:" format
    if "Answer:" in text or "answer:" in text:
        for line in text.split('\n'):
            if "answer" in line.lower() and ":" in line:
                answer = line.split(":")[-1].strip()
                if answer:
                    return answer

    # Return last number-like token
    tokens = text.split()
    for token in reversed(tokens):
        token = token.strip('.,;:!?')
        if token and (token[0].isdigit() or token[0] in '.-'):
            return token

    return ""


def eval_math500(
    model_name: str = "ByteDance/Ouro-1.4B",
    model_type: str = "ouro",
    device: str = "cuda:0",
    max_samples: int = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
):
    """
    Evaluate a model on MATH-500 (mathematical reasoning).

    Args:
        model_name: HuggingFace model ID or local checkpoint path
        model_type: "loopformer" or "ouro" (or any HF causal LM)
        device: torch device
        max_samples: limit evaluation to N samples (None = all 500)
        max_new_tokens: max tokens per generation
        temperature: sampling temperature

    Returns:
        accuracy: float (0-1)
        results: list of dicts with problem, solution, answer, predicted_answer, correct
    """
    print(f"Loading {model_type} model {model_name}...")
    if model_type == "loopformer":
        model, tokenizer = load_model_loopformer(model_name, device)
        model.eval()
        is_loopformer = True
    else:
        model, tokenizer = load_model_ouro(model_name, device)
        model.eval()
        is_loopformer = False

    if not is_loopformer:
        device = next(model.parameters()).device

    print(f"Loading MATH-500...")
    dataset_dict = load_dataset("HuggingFaceH4/MATH-500")
    dataset = dataset_dict["test"]  # or use the only split available
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    results = []

    print(f"Evaluating {len(dataset)} problems...")
    for i, example in enumerate(dataset):
        problem = example["problem"]
        solution = example["solution"]
        level = example["level"]

        # Generate answer
        prompt = f"Solve this math problem:\n\n{problem}\n\nSolution:"
        if is_loopformer:
            prompt_tokens = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
        else:
            prompt_tokens = tokenizer.encode(prompt)

        x = torch.tensor(prompt_tokens, dtype=torch.long, device=device)[None, ...]

        # Generate
        with torch.no_grad():
            if is_loopformer:
                generated = model.generate(
                    x,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=200,
                )
            else:
                outputs = model.generate(
                    x,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=200,
                )
                generated = outputs

        # Decode
        generated_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        generated_answer = generated_text[len(prompt):].strip()

        # Extract ground truth answer (from "answer" field)
        true_answer = example.get("answer", "").strip()

        # Extract predicted answer
        predicted_answer = extract_answer(generated_answer)

        # Check correctness (simple string match, can be improved)
        is_correct = predicted_answer.lower() == true_answer.lower()
        if is_correct:
            correct += 1

        results.append(
            {
                "problem": problem,
                "level": level,
                "solution": solution,
                "generated_answer": generated_answer,
                "predicted_answer": predicted_answer,
                "true_answer": true_answer,
                "correct": is_correct,
            }
        )

        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i + 1}/{len(dataset)}...")

    accuracy = correct / len(results) if results else 0.0
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{len(results)})")

    return accuracy, results


if __name__ == "__main__":
    import fire
    fire.Fire(eval_math500, serialize=lambda x: "")
