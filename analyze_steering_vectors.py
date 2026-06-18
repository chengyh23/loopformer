"""
Analyze steering vector directions across UT steps.
Research question: Do steering vectors maintain consistent directions across loops?

Usage:
  Single trait:   python analyze_steering_vectors.py --trait evil
  All 7 traits:   python analyze_steering_vectors.py --all-traits
  With extraction: python analyze_steering_vectors.py --all-traits --extract
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
import subprocess
from typing import Dict, Optional

TRAITS = ["evil", "apathetic", "hallucinating", "humorous", "impolite", "optimistic", "sycophantic"]
SAVE_DIR = "persona_vectors/Ouro-1.4B"
LOG_DIR = ".log/analyze_steering_vectors"
NUM_UT_STEPS = 4
NUM_LAYERS = 24

def load_steering_vectors(save_dir, trait, num_ut_steps=4, num_layers=24):
    """Load steering vectors for all UT steps and layers."""
    vectors = {}

    for ut_step in range(num_ut_steps):
        prompt_avg = torch.load(f"{save_dir}/{trait}_prompt_avg_diff_ut{ut_step}.pt")
        response_avg = torch.load(f"{save_dir}/{trait}_response_avg_diff_ut{ut_step}.pt")
        prompt_last = torch.load(f"{save_dir}/{trait}_prompt_last_diff_ut{ut_step}.pt")

        vectors[f"prompt_avg_ut{ut_step}"] = prompt_avg
        vectors[f"response_avg_ut{ut_step}"] = response_avg
        vectors[f"prompt_last_ut{ut_step}"] = prompt_last

    return vectors

def compute_cosine_similarities(vectors, num_ut_steps=4, num_layers=24, vector_type="prompt_avg"):
    """Compute cosine similarities between consecutive UT steps for each layer."""
    all_cos_sims = np.zeros((num_layers, num_ut_steps, num_ut_steps))

    for layer in range(num_layers):
        layer_vecs = []
        for ut_step in range(num_ut_steps):
            v = vectors[f"{vector_type}_ut{ut_step}"][layer].numpy()
            layer_vecs.append(v / (np.linalg.norm(v) + 1e-8))

        # Compute pairwise cosine similarities
        for i in range(num_ut_steps):
            for j in range(num_ut_steps):
                all_cos_sims[layer, i, j] = np.dot(layer_vecs[i], layer_vecs[j])

    # Consecutive similarities
    consecutive_cos_sims = np.array([
        all_cos_sims[layer, t, t+1] for layer in range(num_layers) for t in range(num_ut_steps-1)
    ]).reshape(num_layers, num_ut_steps-1)

    return consecutive_cos_sims, all_cos_sims

def analyze_direction_consistency(cos_sims):
    """Analyze if vectors maintain consistent directions."""
    num_layers = cos_sims.shape[0]

    mean_cos = cos_sims.mean(axis=1)
    std_cos = cos_sims.std(axis=1)

    overall_mean = mean_cos.mean()
    overall_std = mean_cos.std()

    return {
        "overall_mean_cosine": float(overall_mean),
        "overall_std_cosine": float(overall_std),
        "per_layer_mean": mean_cos.tolist(),
        "per_layer_std": std_cos.tolist(),
        "min_cosine": float(cos_sims.min()),
        "max_cosine": float(cos_sims.max()),
    }

def ensure_log_dir():
    """Ensure log directory exists."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

def plot_cosine_heatmap(all_cos_sims, save_path, vector_type):
    """Plot heatmap of all pairwise cosine similarities (averaged over layers)."""
    num_ut_steps = all_cos_sims.shape[1]
    avg_cos_sims = all_cos_sims.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(avg_cos_sims, cmap='RdBu_r', vmin=-1, vmax=1)

    ax.set_xticks(range(num_ut_steps))
    ax.set_yticks(range(num_ut_steps))
    ax.set_xticklabels([f"UT{i}" for i in range(num_ut_steps)])
    ax.set_yticklabels([f"UT{i}" for i in range(num_ut_steps)])
    ax.set_xlabel("UT Step")
    ax.set_ylabel("UT Step")
    ax.set_title(f"Cosine Similarity Matrix ({vector_type})\n(averaged across all layers)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cosine Similarity")

    for i in range(num_ut_steps):
        for j in range(num_ut_steps):
            text = ax.text(j, i, f"{avg_cos_sims[i, j]:.3f}",
                          ha="center", va="center", color="black", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_consecutive_similarities(consecutive_cos_sims, save_path, vector_type):
    """Plot consecutive UT step similarities by layer."""
    num_layers = consecutive_cos_sims.shape[0]
    num_transitions = consecutive_cos_sims.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Line plot
    for layer in range(num_layers):
        axes[0].plot(range(num_transitions), consecutive_cos_sims[layer],
                    alpha=0.3, color='blue', linewidth=0.5)

    mean_sims = consecutive_cos_sims.mean(axis=0)
    axes[0].plot(range(num_transitions), mean_sims, color='red', linewidth=2.5,
                label='Mean', marker='o', markersize=8)

    axes[0].set_xlabel("UT Step Transition")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].set_title(f"Direction Consistency Across UT Steps ({vector_type})")
    axes[0].set_xticks(range(num_transitions))
    axes[0].set_xticklabels([f"UT{i}→{i+1}" for i in range(num_transitions)])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_ylim([-0.1, 1.1])

    # Plot 2: Distribution
    axes[1].hist(consecutive_cos_sims.flatten(), bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel("Cosine Similarity")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Distribution of Direction Similarities ({vector_type})")
    axes[1].axvline(mean_sims.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def extract_vectors_for_trait(trait):
    """Extract steering vectors using generate_vec.py script."""
    print(f"\n{'='*70}")
    print(f"EXTRACTING VECTORS FOR TRAIT: {trait.upper()}")
    print(f"{'='*70}")

    model_name = "/home/yc714/proj/decmas/loopformer/Ouro-1.4B"
    model_name_suffix = "Ouro-1.4B"

    result = subprocess.run(
        [
            "python", "generate_vec.py",
            "--model_name", model_name,
            "--pos_path", f"eval_persona_extract/{model_name_suffix}/{trait}_pos_instruct.csv",
            "--neg_path", f"eval_persona_extract/{model_name_suffix}/{trait}_neg_instruct.csv",
            "--trait", trait,
            "--save_dir", f"persona_vectors/{model_name_suffix}/",
            "--threshold", "50",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"ERROR: Failed to extract vectors for {trait}")
        print(result.stderr[-500:] if result.stderr else "")
        return False

    print(f"✓ Successfully extracted vectors for {trait}")
    return True

def analyze_trait(trait):
    """Analyze direction consistency for a single trait."""
    print(f"\n{'='*70}")
    print(f"ANALYZING TRAIT: {trait.upper()}")
    print(f"{'='*70}")

    try:
        vectors = load_steering_vectors(SAVE_DIR, trait, NUM_UT_STEPS, NUM_LAYERS)
        results = {}

        for vector_type in ["prompt_avg", "response_avg", "prompt_last"]:
            consecutive_cos, all_cos = compute_cosine_similarities(
                vectors, NUM_UT_STEPS, NUM_LAYERS, vector_type
            )

            stats = analyze_direction_consistency(consecutive_cos)
            results[vector_type] = {
                "stats": stats,
                "consecutive_similarities": consecutive_cos.tolist(),
                "all_pairwise": all_cos.tolist(),
            }

            print(f"\n{vector_type.upper()}:")
            print(f"  Mean cosine similarity: {stats['overall_mean_cosine']:.4f}")
            print(f"  Std: {stats['overall_std_cosine']:.4f}")

            trait_plot_dir = f"{LOG_DIR}/plots/{trait}"
            Path(trait_plot_dir).mkdir(parents=True, exist_ok=True)

            plot_cosine_heatmap(all_cos, f"{trait_plot_dir}/{vector_type}_heatmap.png", vector_type)
            plot_consecutive_similarities(consecutive_cos, f"{trait_plot_dir}/{vector_type}_consistency.png", vector_type)

        with open(f"{LOG_DIR}/results_{trait}.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Analysis complete for {trait}")
        return results

    except Exception as e:
        print(f"ERROR analyzing {trait}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    ensure_log_dir()
    parser = argparse.ArgumentParser(
        description="Analyze steering vector direction consistency across UT steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single trait:   python analyze_steering_vectors.py --trait evil
  All traits:     python analyze_steering_vectors.py --all-traits
  Extract + analyze: python analyze_steering_vectors.py --all-traits --extract
        """,
    )
    parser.add_argument("--trait", type=str, help="Analyze single trait")
    parser.add_argument("--all-traits", action="store_true", help="Analyze all 7 traits")
    parser.add_argument("--extract", action="store_true", help="Extract vectors before analysis (requires eval_persona data)")

    args = parser.parse_args()

    # Determine which traits to process
    if args.trait:
        traits_to_process = [args.trait]
    elif args.all_traits:
        traits_to_process = TRAITS
    else:
        parser.print_help()
        return

    # Phase 1: Extract if requested
    if args.extract:
        print("\n" + "="*70)
        print("PHASE 1: EXTRACTING STEERING VECTORS")
        print("="*70)

        extraction_results = {}
        for trait in traits_to_process:
            extraction_results[trait] = extract_vectors_for_trait(trait)

        print("\n" + "="*70)
        print("EXTRACTION PHASE SUMMARY")
        print("="*70)
        for trait, success in extraction_results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"  {trait:20s}: {status}")

    # Phase 2: Analyze
    print("\n" + "="*70)
    print("PHASE 2: ANALYZING DIRECTION CONSISTENCY")
    print("="*70)

    all_results = {}
    for trait in traits_to_process:
        all_results[trait] = analyze_trait(trait)

    # Phase 3: Comparative analysis (only if all traits)
    if args.all_traits:
        print("\n" + "="*70)
        print("PHASE 3: COMPARATIVE ANALYSIS ACROSS TRAITS")
        print("="*70)

        comparison = {}
        for vector_type in ["prompt_avg", "response_avg", "prompt_last"]:
            comparison[vector_type] = {}
            for trait in TRAITS:
                if trait in all_results and all_results[trait]:
                    mean_cos = all_results[trait][vector_type]["stats"]["overall_mean_cosine"]
                    comparison[vector_type][trait] = mean_cos

        print("\nMean Cosine Similarity Across Traits:")
        print(f"{'Trait':<20} {'prompt_avg':<15} {'response_avg':<15} {'prompt_last':<15}")
        print("-" * 65)

        for trait in sorted(TRAITS):
            p_avg = comparison["prompt_avg"].get(trait, 0.0)
            r_avg = comparison["response_avg"].get(trait, 0.0)
            p_last = comparison["prompt_last"].get(trait, 0.0)
            print(f"{trait:<20} {p_avg:>14.4f} {r_avg:>14.4f} {p_last:>14.4f}")

        with open(f"{LOG_DIR}/trait_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nGenerated files in {LOG_DIR}/:")
    if args.all_traits:
        print(f"  - trait_comparison.json")
    for trait in traits_to_process:
        print(f"  - results_{trait}.json")
        print(f"  - plots/{trait}/*.png")

if __name__ == "__main__":
    main()
