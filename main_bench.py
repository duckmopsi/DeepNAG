import yaml

from dataset_loader.dataset import Dataset as GestureDataset
from dataset_loader.pipeline import build_dataset_pipeline

import numpy as np
from dataset_loader.transforms import remove_first_dimension
from dataloader.benchmark_adapter import BenchmarkDataset

def prepare_data(dataset, seed=0, single_stroke=False):
    """
    Identical train/validation/test split logic as in torch-diffusion.
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


def main():
    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------
    with open("configs/training_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # -------------------------------------------------------------------------
    # Load raw/canonical dataset
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
    # Apply same class/condition filtering as DDPM
    # -------------------------------------------------------------------------
    dataset = dataset.filter(
        config["class_filters"],
        config["condition_indices"],
    )

    print(f"Loaded gestures: {len(dataset)}")

    # -------------------------------------------------------------------------
    # Same preprocessing pipeline as DDPM
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
    # Same deterministic train/val/test split as DDPM
    # -------------------------------------------------------------------------
    gesture_splits, condition_splits = prepare_data(
        dataset,
        seed=config["seed"],
        single_stroke=True,
    )

    gesture_train, gesture_val, gesture_test = gesture_splits
    condition_train, condition_val, condition_test = condition_splits

    print("Preprocessing finished.")
    print(f"Train gestures: {gesture_train.shape}")
    print(f"Val gestures:   {gesture_val.shape}")
    print(f"Test gestures:  {gesture_test.shape}")

    print(f"Train conditions: {condition_train.shape}")
    print(f"Val conditions:   {condition_val.shape}")
    print(f"Test conditions:  {condition_test.shape}")

    print(f"Example gesture shape: {gesture_train[0].shape}")
    print(f"Example condition: {condition_train[0]}")


    print("\nCondition inspection")
    print("--------------------")
    print(f"class_dims:         {config['class_dims']}")
    print(f"condition_indices:  {config['condition_indices']}")
    print(f"condition_types:    {config['condition_types']}")

    print("\nFirst 5 conditions:")
    for condition in condition_train[:5]:
        print(condition)

if __name__ == "__main__":
    main()