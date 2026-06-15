import json
import os
import sys
import pandas as pd
import asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference import sample


def a_or_an(word):
    return "an" if word[0].lower() in "aeiou" else "a"

def load_persona_questions(trait, persona_instructions_type, assistant_name, version="eval", eval_type="0_100"):
    """
    trait: apathetic, evil, hallucinating, 
    persona_instructions_type: "pos" or "neg"
    assistant_name: 
    version: eval or extract
    
    eval_type: "0_100" o
    """    
    trait_data = json.load(open(f"data_generation/trait_data_{version}/{trait}.json", "r"))
    
    conversations = []
    raw_questions = trait_data["questions"]
    for i, question in enumerate(raw_questions):
        # if i >=1: break # debug
        persona_instructions = [x[persona_instructions_type] for x in trait_data["instruction"]]
        for k, instruction in enumerate(persona_instructions):
            # if k >=1: break # debug
            system = f"You are {a_or_an(assistant_name)} {assistant_name} assistant. {instruction}"
            conversations.append(
                [dict(role='system', content=system)] + [dict(role='user', content=question)]
            )
    print(f"Loaded {len(raw_questions)}x{len(persona_instructions)}={len(conversations)} conversations for trait '{trait}' with instructions type '{persona_instructions_type}' and assistant name '{assistant_name}'.")
    # judge
    from eval.prompts import Prompts
    judge_prompts = {}
    prompt_template = trait_data["eval_prompt"]
    judge_prompts[trait] = prompt_template
    judge_prompts["coherence"] = Prompts[f"coherence_{eval_type}"]
    
    return conversations, judge_prompts

import tiktoken
def load_model_loopformer(model_path: str, device: str = "cuda"):
    # -----------------------------------------------------------------------------
    model_type = "loopformer"
    # init_from = 'armenjeddi/LoopFormer-3block-8iterations-FineWeb300K' # either 'resume' (from an out_dir), 'gpt2' variant, or HuggingFace model ID (e.g. 'armenjeddi/LoopFormer-3block-8iterations-FineWeb300K')
    # -----------------------------------------------------------------------------

    if model_type == "loopformer":
        from models.loopformer import GPTConfig, GPT

    model = GPT.from_pretrained(model_path)
    model.to(device)
    enc = tiktoken.get_encoding("gpt2")
    return model, enc


async def judge_answers(texts, answers, judge_prompts, judge_model="gpt-4o-mini", judge_eval_type="0_100"):
    from judge import OpenAiJudge
    judges = {
        metric: OpenAiJudge(judge_model, prompt, eval_type=judge_eval_type if metric != "coherence" else "0_100")
        for metric, prompt in judge_prompts.items()
    }
    judge_outputs = {}
    for metric, judge in judges.items():
        scores = await asyncio.gather(*[
            judge(question=question, answer=answer)
            for question, answer in zip(texts, answers)
        ])
        judge_outputs[metric] = scores

    return judge_outputs

def main(
    device: str = "cuda:0",
    model_name: str = "armenjeddi/LoopFormer-3block-8iterations-FineWeb300K",
    trait: str = "evil",
    persona_instruction_type: str = "pos",
    assistant_name: str = "evil",
    output_path: str = "eval_persona_extract/LoopFormer-3block-8iterations-FineWeb300K/evil_pos_instruct.csv",
    version: str = "eval",
    overwrite: bool = False,
):
    if os.path.exists(output_path) and not overwrite:
        print(f"Output path {output_path} already exists, skipping...")
        df = pd.read_csv(output_path)
        for trait in [trait , "coherence"]:
            threshold = 50
            print(f"{trait}:  {df[trait].mean():.2f} +- {df[trait].std():.2f}")
        return
    
    model, enc = load_model_loopformer(model_name, device)
    # TODO n_per_question
    conversations, judge_prompts = load_persona_questions(
        trait, persona_instruction_type, assistant_name, version=version)
    # TODO: batch_process
    texts, answers = sample(model, enc, conversations, device=device)

    print(f"Generated answers for questions, now evaluating with judges...")
    outputs = pd.DataFrame(
        [dict(prompt=text, answer=answer) for text, answer in zip(texts, answers)]
    )
    judge_outputs = asyncio.run(judge_answers(texts, answers, judge_prompts))
    for metric, scores in judge_outputs.items():
        outputs[metric] = scores
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    outputs.to_csv(output_path, index=False)
    print(f"Saved evaluation results to {output_path}.")

if __name__ == "__main__":
    import fire
    fire.Fire(main)