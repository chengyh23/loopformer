#!/bin/bash
# Full pipeline: evaluate MMLU → extract accuracy vectors → analyze convergence
# Usage: bash scripts/generate_vec_loopedlm_accuracy.sh [gpu] [num_samples] [num_ut_steps]

set -e

gpu=${1:-0}
num_samples=${2:-50}
num_ut_steps=${3:-8}

echo "=========================================="
echo "ACCURACY STEERING VECTOR PIPELINE"
echo "=========================================="
echo "GPU: $gpu"
echo "Samples: $num_samples"
echo "UT Steps: $num_ut_steps"
echo ""

model_name="/home/yc714/proj/decmas/loopformer/Ouro-1.4B"
if [ "$num_ut_steps" -eq 8 ]; then
    model_name="/home/yc714/proj/decmas/loopformer/Ouro-1.4B-8steps"
fi

# Step 1: Evaluate MMLU and collect completions
echo "Step 1: Evaluating MMLU and saving completions..."
python eval/eval_mmlu.py \
    --model_name "$model_name" \
    --num_samples "$num_samples" \
    --save_dir mmlu_activations

# Step 2: Extract accuracy steering vectors
echo ""
echo "Step 2: Extracting accuracy steering vectors..."
CUDA_VISIBLE_DEVICES=$gpu python generate_vec_accuracy.py \
    --model_name "$model_name" \
    --correct_path mmlu_activations/correct_completions.json \
    --incorrect_path mmlu_activations/incorrect_completions.json \
    --save_dir persona_vectors/Ouro-1.4B/ \
    --num_ut_steps "$num_ut_steps"

echo ""
echo "=========================================="
echo "✓ Pipeline complete!"
echo "=========================================="
echo "Outputs:"
echo "  Completions: mmlu_activations/"
echo "  Vectors: persona_vectors/Ouro-1.4B/accuracy_*_ut*.pt"
echo ""
