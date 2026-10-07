import sys
import yaml
import numpy as np
import torch

from options import Options

from dataset_loader.dataset import Dataset as GestureDataset
from dataset_loader.pipeline import build_dataset_pipeline
from dataset_loader.transforms import remove_first_dimension

from dataloader.benchmark_adapter import BenchmarkDataset


def prepare_data(dataset, seed=0, single_stroke=False):
    """
    Same deterministic train/validation/test split as used in torch-diffusion.
    """
    data = np.asarray(dataset.get_gestures())
    cond = np.asarray(dataset.get_conditions(ohe=True, flatten=True))

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


def main():
    # -------------------------------------------------------------------------
    # DeepGAN options
    # -------------------------------------------------------------------------
    #
    # Keep the original DeepGAN option handling so that model hyperparameters,
    # batch size, latent dimension, etc. continue to come from the original
    # implementation.
    #
    opt = Options()
    opt.parse()

    # -------------------------------------------------------------------------
    # Benchmark preprocessing config
    # -------------------------------------------------------------------------
    with open("configs/training_config.yaml", "r") as f:
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
    # Apply same filtering as DDPM
    # -------------------------------------------------------------------------
    dataset = dataset.filter(
        config["class_filters"],
        config["condition_indices"],
    )

    print(f"Loaded gestures: {len(dataset)}")

    # -------------------------------------------------------------------------
    # Apply same preprocessing pipeline as DDPM
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
    # Same deterministic train/validation/test split as DDPM
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

    print(f"Train conditions: {condition_train.shape}")
    print(f"Val conditions:   {condition_val.shape}")
    print(f"Test conditions:  {condition_test.shape}")

    print(f"Example gesture shape: {gesture_train[0].shape}")
    print(f"Example condition:     {condition_train[0]}")

    # -------------------------------------------------------------------------
    # Inspect conditions
    # -------------------------------------------------------------------------
    print("\nCondition inspection")
    print("--------------------")
    print(f"class_dims:        {config['class_dims']}")
    print(f"condition_indices: {config['condition_indices']}")
    print(f"condition_types:   {config['condition_types']}")

    print("\nFirst 5 conditions:")
    for condition in condition_train[:5]:
        print(condition)

    # -------------------------------------------------------------------------
    # Convert DDPM one-hot conditioning to DeepGAN scalar class labels
    # -------------------------------------------------------------------------
    train_labels = extract_class_labels(condition_train)
    val_labels = extract_class_labels(condition_val)
    test_labels = extract_class_labels(condition_test)

    print("\nExtracted labels")
    print("----------------")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Val labels shape:   {val_labels.shape}")
    print(f"Test labels shape:  {test_labels.shape}")

    print(f"Train classes: {np.unique(train_labels)}")

    print("\nFirst 10 train labels:")
    print(train_labels[:10])

    # -------------------------------------------------------------------------
    # Sanity checks before handing data to DeepGAN
    # -------------------------------------------------------------------------
    if len(gesture_train) != len(train_labels):
        raise RuntimeError(
            "Number of training gestures does not match number of labels."
        )

    if not np.isfinite(gesture_train).all():
        raise RuntimeError(
            "Training gestures contain NaN or Inf values."
        )

    if gesture_train.ndim != 3:
        raise RuntimeError(
            f"Expected training data with shape "
            f"(samples, timesteps, features), got {gesture_train.shape}"
        )

    # -------------------------------------------------------------------------
    # Build adapter dataset for DeepGAN
    # -------------------------------------------------------------------------
    deepgan_dataset = BenchmarkDataset(
        opt=opt,
        gestures=gesture_train,
        labels=train_labels,
    )

    # -------------------------------------------------------------------------
    # Inspect resulting DeepGAN dataset
    # -------------------------------------------------------------------------
    print("\nDeepGAN benchmark dataset")
    print("-------------------------")
    print(f"Samples:      {len(deepgan_dataset.samples)}")
    print(f"Classes:      {deepgan_dataset.num_classes}")
    print(f"Features:     {deepgan_dataset.num_features}")
    print(f"Sequence len: {deepgan_dataset.samples[0].x.shape[0]}")
    print(f"Sample shape: {deepgan_dataset.samples[0].x.shape}")
    print(f"Class map:    {deepgan_dataset.idx_to_class}")

    print("\nFirst DeepGAN sample")
    print("--------------------")
    print(f"x shape: {deepgan_dataset.samples[0].x.shape}")
    print(f"label:   {deepgan_dataset.samples[0].label}")
    print(f"y:       {deepgan_dataset.samples[0].y}")

    # -------------------------------------------------------------------------
    # Test DeepGAN's existing DataSplit + TorchDataLoader
    #
    # BenchmarkDataset.get_split() uses IdentityNormalizer.
    # BenchmarkSample.to_torch() does NOT resample again.
    # -------------------------------------------------------------------------
    data_split = deepgan_dataset.get_split()
    data_loader = data_split.get_data_loader()

    train_loader = data_loader["train"]

    batch_x, batch_y, batch_ids = next(iter(train_loader))

    print("\nDeepGAN DataLoader test")
    print("-----------------------")
    print(f"Batch x shape: {batch_x.shape}")
    print(f"Batch y shape: {batch_y.shape}")
    print(f"Batch x dtype: {batch_x.dtype}")
    print(f"Batch y dtype: {batch_y.dtype}")

    print(f"Batch x min: {batch_x.min().item():.6f}")
    print(f"Batch x max: {batch_x.max().item():.6f}")

    print(f"Batch labels: {batch_y[:10]}")

    print("\nAdapter test successful.")
    print("Data is ready to be passed to DeepGAN.")


if __name__ == "__main__":
    # Same basic Python version check as the original repository.
    if sys.version_info[0] < 3:
        raise Exception(
            "Python 3 or a more recent version is required."
        )

    main()