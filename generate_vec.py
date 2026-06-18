from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
import os
import argparse


def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]
    

def get_hidden_p_and_r(model, tokenizer, prompts, responses, layer_list=None):
    num_hidden_layers = model.config.num_hidden_layers
    total_ut_steps = model.config.total_ut_steps

    if layer_list is None:
        layer_list = list(range(num_hidden_layers))

    # Structure: [ut_step][layer] -> list of activations
    prompt_avg = [[[] for _ in range(num_hidden_layers)] for _ in range(total_ut_steps)]
    response_avg = [[[] for _ in range(num_hidden_layers)] for _ in range(total_ut_steps)]
    prompt_last = [[[] for _ in range(num_hidden_layers)] for _ in range(total_ut_steps)]

    texts = [p+a for p, a in zip(prompts, responses)]
    for text, prompt in tqdm(zip(texts, prompts), total=len(texts)):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for ut_step in range(total_ut_steps):
            for layer in layer_list:
                hidden_idx = 1 + ut_step * num_hidden_layers + layer
                hidden = outputs.hidden_states[hidden_idx]

                prompt_avg[ut_step][layer].append(hidden[:, :prompt_len, :].mean(dim=1).detach().cpu())
                response_avg[ut_step][layer].append(hidden[:, prompt_len:, :].mean(dim=1).detach().cpu())
                prompt_last[ut_step][layer].append(hidden[:, prompt_len-1, :].detach().cpu())
        del outputs

    for ut_step in range(total_ut_steps):
        for layer in layer_list:
            if prompt_avg[ut_step][layer]:
                prompt_avg[ut_step][layer] = torch.cat(prompt_avg[ut_step][layer], dim=0)
            if prompt_last[ut_step][layer]:
                prompt_last[ut_step][layer] = torch.cat(prompt_last[ut_step][layer], dim=0)
            if response_avg[ut_step][layer]:
                response_avg[ut_step][layer] = torch.cat(response_avg[ut_step][layer], dim=0)

    return prompt_avg, prompt_last, response_avg

import pandas as pd
import os

def get_persona_effective(pos_path, neg_path, trait, threshold=50):
    persona_pos = pd.read_csv(pos_path)
    persona_neg = pd.read_csv(neg_path)
    mask = (persona_pos[trait] >=threshold) & (persona_neg[trait] < 100-threshold) & (persona_pos["coherence"] >= 50) & (persona_neg["coherence"] >= 50)

    persona_pos_effective = persona_pos[mask]
    persona_neg_effective = persona_neg[mask]

    persona_pos_effective_prompts = persona_pos_effective["prompt"].tolist()    
    persona_neg_effective_prompts = persona_neg_effective["prompt"].tolist()

    persona_pos_effective_responses = persona_pos_effective["answer"].tolist()
    persona_neg_effective_responses = persona_neg_effective["answer"].tolist()

    return persona_pos_effective, persona_neg_effective, persona_pos_effective_prompts, persona_neg_effective_prompts, persona_pos_effective_responses, persona_neg_effective_responses


def save_persona_vector(model_name, pos_path, neg_path, trait, save_dir, threshold=50):
    # Check if vectors already exist
    from pathlib import Path
    save_dir_path = Path(save_dir)

    # Load model config to get num_ut_steps without loading full model
    import json
    config_path = os.path.join(model_name, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    num_ut_steps = config.get("total_ut_steps", 1)

    # Check if all output files exist
    required_files = []
    for ut_step in range(num_ut_steps):
        required_files.extend([
            f"{trait}_prompt_avg_diff_ut{ut_step}.pt",
            f"{trait}_response_avg_diff_ut{ut_step}.pt",
            f"{trait}_prompt_last_diff_ut{ut_step}.pt",
        ])

    all_exist = all((save_dir_path / f).exists() for f in required_files)
    if all_exist:
        print(f"✓ Steering vectors for '{trait}' already exist. Skipping extraction.")
        return

    print(f"🔄 Extracting steering vectors for '{trait}'...")

    # TODO: when to trust_remote_code=True? (for now, for loopformer and Ouro)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    _, _, persona_pos_effective_prompts, persona_neg_effective_prompts, persona_pos_effective_responses, persona_neg_effective_responses = get_persona_effective(pos_path, neg_path, trait, threshold)

    persona_effective_prompt_avg, persona_effective_prompt_last, persona_effective_response_avg = {}, {}, {}

    persona_effective_prompt_avg["pos"], persona_effective_prompt_last["pos"], persona_effective_response_avg["pos"] = get_hidden_p_and_r(model, tokenizer, persona_pos_effective_prompts, persona_pos_effective_responses)
    persona_effective_prompt_avg["neg"], persona_effective_prompt_last["neg"], persona_effective_response_avg["neg"] = get_hidden_p_and_r(model, tokenizer, persona_neg_effective_prompts, persona_neg_effective_responses)

    total_ut_steps = model.config.total_ut_steps
    num_hidden_layers = model.config.num_hidden_layers

    os.makedirs(save_dir, exist_ok=True)

    for ut_step in range(total_ut_steps):
        prompt_avg_diff = torch.stack(
            [persona_effective_prompt_avg["pos"][ut_step][l].mean(0).float() -
             persona_effective_prompt_avg["neg"][ut_step][l].mean(0).float()
             for l in range(num_hidden_layers)],
            dim=0
        )
        response_avg_diff = torch.stack(
            [persona_effective_response_avg["pos"][ut_step][l].mean(0).float() -
             persona_effective_response_avg["neg"][ut_step][l].mean(0).float()
             for l in range(num_hidden_layers)],
            dim=0
        )
        prompt_last_diff = torch.stack(
            [persona_effective_prompt_last["pos"][ut_step][l].mean(0).float() -
             persona_effective_prompt_last["neg"][ut_step][l].mean(0).float()
             for l in range(num_hidden_layers)],
            dim=0
        )

        torch.save(prompt_avg_diff, f"{save_dir}/{trait}_prompt_avg_diff_ut{ut_step}.pt")
        torch.save(response_avg_diff, f"{save_dir}/{trait}_response_avg_diff_ut{ut_step}.pt")
        torch.save(prompt_last_diff, f"{save_dir}/{trait}_prompt_last_diff_ut{ut_step}.pt")

    print(f"Persona vectors saved to {save_dir}")    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--pos_path", type=str, required=True)
    parser.add_argument("--neg_path", type=str, required=True)
    parser.add_argument("--trait", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--threshold", type=int, default=50)
    args = parser.parse_args()
    save_persona_vector(args.model_name, args.pos_path, args.neg_path, args.trait, args.save_dir, args.threshold)