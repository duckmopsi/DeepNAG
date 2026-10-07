import sys
import argparse
import yaml
import numpy as np
import torch

from options import Options
from models.DeepGAN import DeepGAN

from dataset_loader.dataset import Dataset as GestureDataset
from dataset_loader.pipeline import build_dataset_pipeline
from dataset_loader.transforms import remove_first_dimension

from dataloader.benchmark_adapter import BenchmarkDataset


def parse_benchmark_config_argument():
    """
    Parse only the benchmark-specific config argument.

    All remaining command-line arguments are left for DeepNAG/DeepGAN's
    original Options parser.
    """
    parser = argparse.ArgumentParser(
        add_help=False
    )

    parser.add_argument(
        "--benchmark-config",
        type=str,
        required=True,
        help="Path to the benchmark preprocessing YAML config.",
    )

    args, remaining_args = parser.parse_known_args()

    # Remove our custom argument before the original DeepGAN Options parser
    # sees sys.argv.
    sys.argv = [sys.argv[0]] + remaining_args

    return args.benchmark_config


def prepare_data(dataset, seed=0, single_stroke=False):
    """
    Same deterministic train/validation/test split as used in torch-diffusion.
    """
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

    data_splits = (
        data[:train_end],
        data[train_end:val_end],
        data[val_end:],
    )

    condition_splits = (
        cond[:train_end],
        cond[train_end:val_end],
        cond[val_end:],
    )

    return data_splits, condition_splits


def extract_class_labels(conditions):
    """
    Convert one-hot encoded gesture conditions to scalar class labels.

    Example:
        [1, 0, 0, 0, 0] -> 0
        [0, 0, 0, 1, 0] -> 3
    """
    if conditions.ndim != 2:
        raise ValueError(
            f"Expected 2D condition array, got shape {conditions.shape}"
        )

    return np.argmax(conditions, axis=1)


def train(model_t, dataset, device):
    """
    Original DeepGAN training entry point.
    """
    data_split = dataset.get_split()

    model = model_t(
        dataset.num_classes,
        dataset.num_features,
        dataset.opt,
        device,
        dataset.visualizer,
    )

    model.run_training_loop(data_split)
    model.save()


def main():
    # -------------------------------------------------------------------------
    # Parse benchmark-specific argument first
    # -------------------------------------------------------------------------
    benchmark_config_path = parse_benchmark_config_argument()

    print(
        f"Using benchmark config: "
        f"{benchmark_config_path}"
    )

    # -------------------------------------------------------------------------
    # Original DeepGAN options
    # -------------------------------------------------------------------------
    #
    # At this point --benchmark-config has already been removed from sys.argv.
    # The remaining arguments are parsed by the original implementation.
    #
    opt = Options()
    opt.parse()

    # -------------------------------------------------------------------------
    # Shared preprocessing config
    # -------------------------------------------------------------------------
    with open(
        benchmark_config_path,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    # -------------------------------------------------------------------------
    # Load canonical dataset
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Same filtering as DDPM
    # -------------------------------------------------------------------------
    dataset = dataset.filter(
        config["class_filters"],
        config["condition_indices"],
    )

    print(f"Loaded gestures: {len(dataset)}")

    # -------------------------------------------------------------------------
    # Same preprocessing as DDPM
    # -------------------------------------------------------------------------
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
    )

    # -------------------------------------------------------------------------
    # Same deterministic split as DDPM
    # -------------------------------------------------------------------------
    gesture_splits, condition_splits = prepare_data(
        dataset,
        seed=config["seed"],
        single_stroke=True,
    )

    gesture_train, gesture_val, gesture_test = gesture_splits
    condition_train, condition_val, condition_test = condition_splits

    print("\nPreprocessing finished.")
    print("-----------------------")
    print(f"Train gestures: {gesture_train.shape}")
    print(f"Val gestures:   {gesture_val.shape}")
    print(f"Test gestures:  {gesture_test.shape}")

    print(
        f"Train conditions: "
        f"{condition_train.shape}"
    )
    print(
        f"Val conditions:   "
        f"{condition_val.shape}"
    )
    print(
        f"Test conditions:  "
        f"{condition_test.shape}"
    )

    # -------------------------------------------------------------------------
    # Fixed-dt benchmark sanity checks
    # -------------------------------------------------------------------------
    #
    # In interpolate mode, t is implicit:
    #
    #     t_i = i * dt
    #
    # Therefore DeepGAN should only receive x/y.
    #
    if config["mode"] != "interpolate":
        raise RuntimeError(
            "This benchmark run currently expects "
            "mode='interpolate' for the fixed-dt experiment."
        )

    if gesture_train.ndim != 3:
        raise RuntimeError(
            "Expected gesture array with shape "
            "(samples, timesteps, features), "
            f"got {gesture_train.shape}."
        )

    if gesture_train.shape[2] != 2:
        raise RuntimeError(
            "Fixed-dt DeepGAN benchmark expects exactly "
            "2 features (x, y), but got "
            f"{gesture_train.shape[2]}."
        )

    if not np.isfinite(gesture_train).all():
        raise RuntimeError(
            "Training gestures contain NaN or Inf values."
        )

    # -------------------------------------------------------------------------
    # Convert DDPM conditions to DeepGAN classes
    # -------------------------------------------------------------------------
    train_labels = extract_class_labels(
        condition_train
    )
    val_labels = extract_class_labels(
        condition_val
    )
    test_labels = extract_class_labels(
        condition_test
    )

    print("\nExtracted labels")
    print("----------------")
    print(
        f"Train classes: "
        f"{np.unique(train_labels)}"
    )
    print(
        f"Val classes:   "
        f"{np.unique(val_labels)}"
    )
    print(
        f"Test classes:  "
        f"{np.unique(test_labels)}"
    )

    # -------------------------------------------------------------------------
    # Synchronize sequence length
    # -------------------------------------------------------------------------
    sequence_length = gesture_train.shape[1]

    # DeepGAN uses resample_n as the generated sequence length.
    # Our BenchmarkSample does NOT resample the input again.
    opt.resample_n = sequence_length

    print("\nSequence configuration")
    print("----------------------")
    print(
        f"Preprocessing mode:    "
        f"{config['mode']}"
    )
    print(
        f"Fixed timestep:        "
        f"{config['dt']} s"
    )
    print(
        f"Sequence length:       "
        f"{sequence_length}"
    )
    print(
        f"DeepGAN resample_n:    "
        f"{opt.resample_n}"
    )

    # -------------------------------------------------------------------------
    # Build DeepGAN dataset adapter
    # -------------------------------------------------------------------------
    deepgan_dataset = BenchmarkDataset(
        opt=opt,
        gestures=gesture_train,
        labels=train_labels,
    )

    print("\nDeepGAN benchmark dataset")
    print("-------------------------")
    print(
        f"Samples:      "
        f"{len(deepgan_dataset.samples)}"
    )
    print(
        f"Classes:      "
        f"{deepgan_dataset.num_classes}"
    )
    print(
        f"Features:     "
        f"{deepgan_dataset.num_features}"
    )
    print(
        f"Sequence len: "
        f"{deepgan_dataset.samples[0].x.shape[0]}"
    )
    print(
        f"Sample shape: "
        f"{deepgan_dataset.samples[0].x.shape}"
    )
    print(
        f"Class map:    "
        f"{deepgan_dataset.idx_to_class}"
    )

    # -------------------------------------------------------------------------
    # DataLoader sanity check
    # -------------------------------------------------------------------------
    data_split = deepgan_dataset.get_split()
    data_loader = data_split.get_data_loader()

    train_loader = data_loader["train"]

    batch_x, batch_y, batch_ids = next(
        iter(train_loader)
    )

    print("\nDeepGAN DataLoader sanity check")
    print("-------------------------------")
    print(
        f"Batch x shape: "
        f"{batch_x.shape}"
    )
    print(
        f"Batch y shape: "
        f"{batch_y.shape}"
    )
    print(
        f"Batch x dtype: "
        f"{batch_x.dtype}"
    )
    print(
        f"Batch y dtype: "
        f"{batch_y.dtype}"
    )
    print(
        f"Batch x min:   "
        f"{batch_x.min().item():.6f}"
    )
    print(
        f"Batch x max:   "
        f"{batch_x.max().item():.6f}"
    )
    print(
        f"Batch labels:  "
        f"{batch_y[:10]}"
    )

    if batch_x.shape[1] != sequence_length:
        raise RuntimeError(
            "Unexpected DeepGAN sequence length: "
            f"{batch_x.shape[1]} instead of "
            f"{sequence_length}."
        )

    if batch_x.shape[2] != 2:
        raise RuntimeError(
            "DeepGAN received an unexpected feature "
            f"dimension: {batch_x.shape[2]}."
        )

    print("\nData pipeline sanity check passed.")

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------
    if opt.use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        device = torch.device("cuda:0")

        print("\nUsing CUDA")
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:
        device = torch.device("cpu")

        print("\nUsing CPU")

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    print("\nStarting DeepGAN training")
    print("-------------------------")

    train(
        DeepGAN,
        deepgan_dataset,
        device,
    )


if __name__ == "__main__":
    if sys.version_info[0] < 3:
        raise Exception(
            "Python 3 or newer is required."
        )

    main()