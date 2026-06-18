set -e
gpu=${1:-0}
# model_type="loopformer"
# model_name="armenjeddi/LoopFormer-3block-8iterations-FineWeb300K"
model_type="ouro"
# model_name="ByteDance/Ouro-1.4B"
model_name="/home/yc714/proj/decmas/loopformer/Ouro-1.4B"
model_name_suffix="${model_name##*/}"

TRAITS=("evil" "apathetic" "hallucinating" "humorous" "impolite" "optimistic" "sycophantic")
for trait in "${TRAITS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing trait: $trait"
    echo "=========================================="

    echo "🔄 Generating positive instructions ..."
    CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
        --trait $trait \
        --model_type $model_type \
        --model_name $model_name \
        --output_path eval_persona_extract/$model_name_suffix/${trait}_pos_instruct.csv \
        --persona_instruction_type pos \
        --assistant_name $trait \
        --version extract
        # --batch_process False \

    echo "🔄 Generating negative instructions ..."
    CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
        --trait $trait \
        --output_path eval_persona_extract/$model_name_suffix/${trait}_neg_instruct.csv \
        --persona_instruction_type neg \
        --assistant_name helpful \
        --version extract

    echo "🔄 Generating persona vectors ..."
    CUDA_VISIBLE_DEVICES=$gpu python generate_vec.py \
        --model_name $model_name \
        --pos_path eval_persona_extract/$model_name_suffix/${trait}_pos_instruct.csv \
        --neg_path eval_persona_extract/$model_name_suffix/${trait}_neg_instruct.csv \
        --trait $trait \
        --save_dir persona_vectors/$model_name_suffix/ \
        --threshold 50
    
done
echo ""
echo "All traits processed!"