set -e
gpu=${1:-0}


echo "Generating positive instructions ..."
CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
    --trait evil \
    --output_path eval_persona_extract/LoopFormer-3block-8iterations-FineWeb300K/evil_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name evil \
    --version extract
    # --batch_process False \

echo "Generating negative instructions ..."
CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
    --trait evil \
    --output_path eval_persona_extract/LoopFormer-3block-8iterations-FineWeb300K/evil_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --version extract


CUDA_VISIBLE_DEVICES=$gpu python generate_vec.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --pos_path eval_persona_extract/LoopFormer-3block-8iterations-FineWeb300K/evil_pos_instruct.csv \
    --neg_path eval_persona_extract/LoopFormer-3block-8iterations-FineWeb300K/evil_neg_instruct.csv \
    --trait evil \
    --save_dir persona_vectors/LoopFormer-3block-8iterations-FineWeb300K/ \
    --threshold 50