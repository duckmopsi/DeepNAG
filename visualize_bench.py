import argparse
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from dataset_loader.dataset import Dataset as GestureDataset
from dataset_loader.pipeline import build_dataset_pipeline
from dataset_loader.transforms import remove_first_dimension

from models.DeepGAN import DeepGAN


# =============================================================================
# Arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and visualize samples from a trained DeepGAN benchmark."
    )

    parser.add_argument(
        "--benchmark-config",
        required=True,
        type=str,
        help="Dataset/preprocessing config used for training.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to DeepGAN checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        default="generated",
        type=str,
        help="Directory for generated figures and arrays.",
    )

    parser.add_argument(
        "--samples-per-class",
        default=20,
        type=int,
        help="Number of generated samples per class.",
    )

    parser.add_argument(
        "--latent-dim",
        default=32,
        type=int,
        help="Latent dimension used during DeepGAN training.",
    )

    parser.add_argument(
        "--seed",
        default=0,
        type=int,
    )

    return parser.parse_args()


# =============================================================================
# Shared benchmark preprocessing
# =============================================================================

def prepare_data(dataset, seed=0, single_stroke=False):
    data = np.asarray(dataset.get_gestures())

    cond = np.asarray(
        dataset.get_conditions(
            ohe=True,
            flatten=True,
        )
    )

    if single_stroke:
        data = remove_first_dimension(data)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(data))

    data = data[idx]
    cond = cond[idx]

    n = len(data)

    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    return (
        (
            data[:train_end],
            data[train_end:val_end],
            data[val_end:],
        ),
        (
            cond[:train_end],
            cond[train_end:val_end],
            cond[val_end:],
        ),
    )


def extract_class_labels(conditions):
    return np.argmax(
        conditions,
        axis=1,
    )


def load_benchmark_data(config):
    dataset = GestureDataset.from_json(
        config["data_path"],
        config["interpolated_dt"],
        False,
        False,
        config["class_dims"],
        config["condition_types"],
        config["min_size"],
        config["max_size"],
        config["min_strokes"],
        config["max_strokes"],
    )

    dataset = dataset.filter(
        config["class_filters"],
        config["condition_indices"],
    )

    dataset = build_dataset_pipeline(
        dataset,
        representation=config["representation"],
        mode=config["mode"],
        num_points=config["num_points"],
        dt=config["dt"],
        normalize=config["normalize"],
        pos_bounds=config.get("pos_bounds"),
        velo_bounds=config.get("velo_bounds"),
        pad_value=config["pad_value"],
        min_size=config.get("min_size"),
        max_size=config.get("max_size"),
    )

    gesture_splits, condition_splits = prepare_data(
        dataset,
        seed=config["seed"],
        single_stroke=True,
    )

    return gesture_splits, condition_splits


# =============================================================================
# DeepGAN
# =============================================================================

def create_deepgan_options(
    sequence_length,
    latent_dim,
    seed,
):
    """
    Minimal options object containing everything required to construct
    DeepGAN for inference.
    """
    return SimpleNamespace(
        seed=seed,

        latent_dim=latent_dim,
        resample_n=sequence_length,

        # Required when constructing the optimizers. Their states are
        # replaced by the checkpoint afterwards.
        lr=1e-4,
        beta0=0.5,
        beta1=0.9,

        batch_size=64,

        deepgan_critic_iters=5,
        deepgan_lambda=10,

        use_tensorboard=0,
        run_tb_dir=None,
    )


def load_checkpoint(model, checkpoint, device):
    print(f"Loading checkpoint: {checkpoint}")

    # weights_only=False is needed for checkpoints containing the
    # IdentityNormalizer object on newer PyTorch versions.
    try:
        state = torch.load(
            checkpoint,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        state = torch.load(
            checkpoint,
            map_location=device,
        )

    model._load_state(state)

    model._generator.eval()
    model._discriminator.eval()


def generate_samples(
    model,
    class_idx,
    count,
    device,
):
    labels = torch.full(
        (count,),
        class_idx,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        generated = model.generate(
            labels,
            unnormalize=False,
        )

    return generated.detach().cpu().numpy()


# =============================================================================
# Normalization
# =============================================================================

def unnormalize_positions(data, pos_bounds):
    """
    Reverse the shared position normalization.

    pos_bounds:
        [source_min, source_max, target_min, target_max]

    Example:
        [0, 1, -1, 1]
    """
    d_min, d_max, i_min, i_max = pos_bounds

    return (
        (data - i_min)
        / (i_max - i_min)
        * (d_max - d_min)
        + d_min
    )


# =============================================================================
# Visualization
# =============================================================================

def setup_axis(ax, title=None):
    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 0.0)

    if title is not None:
        ax.set_title(title)


def plot_generated_overlay(
    samples,
    class_idx,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    for sample in samples:
        ax.plot(
            sample[:, 0],
            sample[:, 1],
            alpha=0.45,
            linewidth=1.0,
        )

    setup_axis(
        ax,
        title=f"DeepGAN generated samples - class {class_idx}",
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


def plot_real_vs_generated(
    real_samples,
    generated_samples,
    class_idx,
    output_path,
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6),
    )

    for sample in real_samples:
        axes[0].plot(
            sample[:, 0],
            sample[:, 1],
            alpha=0.4,
            linewidth=1.0,
        )

    for sample in generated_samples:
        axes[1].plot(
            sample[:, 0],
            sample[:, 1],
            alpha=0.4,
            linewidth=1.0,
        )

    setup_axis(
        axes[0],
        title=f"Real - class {class_idx}",
    )

    setup_axis(
        axes[1],
        title=f"Generated - class {class_idx}",
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


def plot_individual_samples(
    samples,
    class_idx,
    output_path,
    max_samples=16,
):
    n = min(
        len(samples),
        max_samples,
    )

    cols = 4
    rows = int(
        np.ceil(n / cols)
    )

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(12, 3 * rows),
    )

    axes = np.asarray(
        axes
    ).reshape(-1)

    for idx in range(n):
        axes[idx].plot(
            samples[idx, :, 0],
            samples[idx, :, 1],
            linewidth=1.5,
        )

        setup_axis(
            axes[idx],
            title=f"Sample {idx + 1}",
        )

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        f"DeepGAN individual samples - class {class_idx}"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


def plot_all_classes(
    generated_by_class,
    output_path,
):
    class_ids = sorted(
        generated_by_class.keys()
    )

    fig, axes = plt.subplots(
        1,
        len(class_ids),
        figsize=(5 * len(class_ids), 5),
    )

    if len(class_ids) == 1:
        axes = [axes]

    for ax, class_idx in zip(
        axes,
        class_ids,
    ):
        samples = generated_by_class[
            class_idx
        ]

        for sample in samples:
            ax.plot(
                sample[:, 0],
                sample[:, 1],
                alpha=0.35,
                linewidth=1.0,
            )

        setup_axis(
            ax,
            title=f"Class {class_idx}",
        )

    fig.suptitle(
        "DeepGAN generated samples"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    torch.manual_seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------

    with open(
        args.benchmark_config,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    # -------------------------------------------------------------------------
    # Load real benchmark data using exactly the same preprocessing
    # -------------------------------------------------------------------------

    gesture_splits, condition_splits = load_benchmark_data(
        config
    )

    gesture_train, _, _ = gesture_splits
    condition_train, _, _ = condition_splits

    train_labels = extract_class_labels(
        condition_train
    )

    sequence_length = gesture_train.shape[1]
    num_features = gesture_train.shape[2]

    class_ids = sorted(
        np.unique(
            train_labels
        ).tolist()
    )

    num_classes = len(
        class_ids
    )

    print("Benchmark data")
    print("--------------")
    print(
        f"Training samples: {len(gesture_train)}"
    )
    print(
        f"Sequence length:  {sequence_length}"
    )
    print(
        f"Features:         {num_features}"
    )
    print(
        f"Classes:          {class_ids}"
    )

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------

    if torch.cuda.is_available():
        device = torch.device(
            "cuda:0"
        )

        print(
            f"Using GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )
    else:
        device = torch.device(
            "cpu"
        )

        print("Using CPU")

    # -------------------------------------------------------------------------
    # DeepGAN
    # -------------------------------------------------------------------------

    opt = create_deepgan_options(
        sequence_length=sequence_length,
        latent_dim=args.latent_dim,
        seed=args.seed,
    )

    model = DeepGAN(
        num_classes=num_classes,
        num_features=num_features,
        opt=opt,
        device=device,
        visualizer=None,
    )

    load_checkpoint(
        model,
        args.checkpoint,
        device,
    )

    # -------------------------------------------------------------------------
    # Generate
    # -------------------------------------------------------------------------

    generated_by_class = {}

    for class_idx in class_ids:

        print(
            f"Generating class {class_idx}..."
        )

        generated = generate_samples(
            model=model,
            class_idx=class_idx,
            count=args.samples_per_class,
            device=device,
        )

        # Convert both generated and real coordinates back from [-1, 1]
        # into the original position range, normally [0, 1].
        if config["normalize"]:
            generated_plot = unnormalize_positions(
                generated,
                config["pos_bounds"],
            )
        else:
            generated_plot = generated

        generated_by_class[
            class_idx
        ] = generated_plot

        # ---------------------------------------------------------------------
        # Corresponding real samples
        # ---------------------------------------------------------------------

        real_indices = np.where(
            train_labels == class_idx
        )[0]

        rng = np.random.RandomState(
            args.seed + class_idx
        )

        real_indices = rng.choice(
            real_indices,
            size=min(
                args.samples_per_class,
                len(real_indices),
            ),
            replace=False,
        )

        real = gesture_train[
            real_indices
        ]

        if config["normalize"]:
            real_plot = unnormalize_positions(
                real,
                config["pos_bounds"],
            )
        else:
            real_plot = real

        # ---------------------------------------------------------------------
        # Save raw generated arrays
        # ---------------------------------------------------------------------

        np.save(
            os.path.join(
                args.output_dir,
                f"class_{class_idx}_generated.npy",
            ),
            generated,
        )

        # ---------------------------------------------------------------------
        # Figures
        # ---------------------------------------------------------------------

        plot_generated_overlay(
            generated_plot,
            class_idx,
            os.path.join(
                args.output_dir,
                f"class_{class_idx}_overlay.png",
            ),
        )

        plot_individual_samples(
            generated_plot,
            class_idx,
            os.path.join(
                args.output_dir,
                f"class_{class_idx}_individual.png",
            ),
        )

        plot_real_vs_generated(
            real_plot,
            generated_plot,
            class_idx,
            os.path.join(
                args.output_dir,
                f"class_{class_idx}_real_vs_generated.png",
            ),
        )

    # -------------------------------------------------------------------------
    # Overview
    # -------------------------------------------------------------------------

    plot_all_classes(
        generated_by_class,
        os.path.join(
            args.output_dir,
            "all_classes.png",
        ),
    )

    print(
        f"\nFinished. Results saved to: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()