from itertools import product
from pathlib import Path
import argparse
import json
import subprocess
import sys


# =============================================================================
# Search space
# =============================================================================

LEARNING_RATES = [
    5e-5,
    1e-4,
    2e-4,
]

BATCH_SIZES = [
    32,
    64,
    128,
]


def build_grid():
    """
    Build the full DeepGAN / DeepNAG hyperparameter grid.
    """
    return [
        {
            "lr": lr,
            "batch_size": batch_size,
        }
        for lr, batch_size in product(
            LEARNING_RATES,
            BATCH_SIZES,
        )
    ]


# =============================================================================
# Configuration
# =============================================================================

def get_benchmark_config(representation):
    if representation == "xy":
        return "configs/deepgan_fixed_dt.yaml"

    if representation == "xyt":
        return "configs/deepgan_free_t.yaml"

    raise ValueError(
        f"Unknown representation: {representation}"
    )


def format_float(value):
    """
    Create a compact representation suitable for experiment names.
    """
    return f"{value:.0e}".replace("-", "m")


def build_experiment_name(
    model,
    representation,
    params,
    seed,
):
    lr = format_float(params["lr"])
    batch_size = params["batch_size"]

    return (
        f"ablation_"
        f"{model.lower()}_"
        f"{representation}_"
        f"lr{lr}_"
        f"bs{batch_size}_"
        f"seed{seed}"
    )


# =============================================================================
# Command construction
# =============================================================================

def build_command(
    model,
    representation,
    params,
    seed,
    epochs,
    benchmark_config,
    checkpoint_frequency,
):
    experiment_name = build_experiment_name(
        model=model,
        representation=representation,
        params=params,
        seed=seed,
    )

    command = [
        sys.executable,
        "-u",
        "main_bench.py",

        "--benchmark-config",
        benchmark_config,

        "--model",
        model,

        "--dataset-name",
        "prolific",

        "--seed",
        str(seed),

        "--epochs",
        str(epochs),

        "--batch-size",
        str(params["batch_size"]),

        "--lr",
        str(params["lr"]),

        "--vis-frequency",
        "0",

        "--checkpoint-frequency",
        str(checkpoint_frequency),

        "--experiment-name",
        experiment_name,
    ]

    # DeepGAN-specific fixed parameters.
    if model == "DeepGAN":
        command.extend([
            "--deepgan-critic-iters",
            "5",

            "--deepgan-lambda",
            "10",
        ])

    return experiment_name, command


# =============================================================================
# Metadata
# =============================================================================

def save_run_config(
    experiment_name,
    model,
    representation,
    benchmark_config,
    params,
    seed,
    epochs,
):
    out_dir = Path("ablation_configs")
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "experiment_name": experiment_name,
        "model": model,
        "representation": representation,
        "benchmark_config": benchmark_config,
        "seed": seed,
        "epochs": epochs,
        "hyperparameters": params,
    }

    path = out_dir / f"{experiment_name}.json"

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    return path


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "DeepGAN",
            "DeepNAG",
        ],
        required=True,
    )

    parser.add_argument(
        "--representation",
        choices=[
            "xy",
            "xyt",
        ],
        required=True,
    )

    parser.add_argument(
        "--index",
        type=int,
        help="Index of the hyperparameter configuration to run.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--benchmark-config",
        type=str,
        default=None,
        help=(
            "Optional preprocessing config. "
            "If omitted, it is inferred from the representation."
        ),
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the full grid and exit.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without starting training.",
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    grid = build_grid()

    if args.list:
        print(
            f"Number of configurations: {len(grid)}"
        )

        for index, params in enumerate(grid):
            print(
                f"{index}: "
                f"lr={params['lr']}, "
                f"batch_size={params['batch_size']}"
            )

        return

    if args.index is None:
        raise ValueError(
            "--index is required unless --list is used."
        )

    if not 0 <= args.index < len(grid):
        raise ValueError(
            f"Invalid index {args.index}. "
            f"Expected 0-{len(grid) - 1}."
        )

    params = grid[args.index]

    benchmark_config = (
        args.benchmark_config
        if args.benchmark_config is not None
        else get_benchmark_config(
            args.representation
        )
    )

    experiment_name, command = build_command(
        model=args.model,
        representation=args.representation,
        params=params,
        seed=args.seed,
        epochs=args.epochs,
        benchmark_config=benchmark_config,
        checkpoint_frequency=args.checkpoint_frequency,
    )

    config_path = save_run_config(
        experiment_name=experiment_name,
        model=args.model,
        representation=args.representation,
        benchmark_config=benchmark_config,
        params=params,
        seed=args.seed,
        epochs=args.epochs,
    )

    print("=" * 70)
    print("Ablation run")
    print("=" * 70)
    print(f"Index:          {args.index}")
    print(f"Model:          {args.model}")
    print(f"Representation: {args.representation}")
    print(f"Learning rate:  {params['lr']}")
    print(f"Batch size:     {params['batch_size']}")
    print(f"Seed:           {args.seed}")
    print(f"Epochs:         {args.epochs}")
    print(f"Experiment:     {experiment_name}")
    print(f"Metadata:       {config_path}")
    print()
    print("Command:")
    print(" ".join(command))
    print("=" * 70)

    if args.dry_run:
        return

    subprocess.run(
        command,
        check=True,
    )


if __name__ == "__main__":
    main()