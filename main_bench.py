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


# =============================================================================
# Command-line arguments specific to benchmark pipeline
# =============================================================================

def parse_benchmark_arguments():
    """
    Parse arguments that belong to our benchmark wrapper.

    Remaining arguments are left untouched for the original DeepGAN
    Options parser.
    """
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--benchmark-config",
        type=str,
        required=True,
        help="Path to benchmark preprocessing YAML config.",
    )

    parser.add_argument(
        "--debug-only",
        action="store_true",
        help="Run preprocessing + adapter checks and stop before training.",
    )

    args, remaining_args = parser.parse_known_args()

    # DeepGAN's original Options parser must not see our custom arguments.
    sys.argv = [sys.argv[0]] + remaining_args

    return args


# =============================================================================
# Dataset debugging helpers
# =============================================================================

def extract_sequence(gesture):
    """
    Try to extract the actual timestep x feature array from a gesture.

    The shared Dataset stores single-stroke gestures with an additional
    stroke dimension in some stages of preprocessing.
    """
    current = gesture

    # Unwrap lists / tuples containing exactly one stroke.
    while isinstance(current, (list, tuple)) and len(current) == 1:
        current = current[0]

    arr = np.asarray(current)

    # Single-stroke representation:
    # (1, timesteps, features) -> (timesteps, features)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    # Object array containing one stroke.
    if arr.dtype == object and arr.ndim == 1 and len(arr) == 1:
        arr = np.asarray(arr[0])

    return arr


def get_sequence_length(gesture):
    arr = extract_sequence(gesture)

    if arr.ndim == 0:
        return 0

    return arr.shape[0]


def describe_gesture(gesture):
    arr = extract_sequence(gesture)

    info = {
        "shape": arr.shape,
        "length": arr.shape[0] if arr.ndim > 0 else 0,
    }

    # If timestamps are still present, report their range.
    if arr.ndim == 2 and arr.shape[1] >= 3:
        t = arr[:, 2]

        if len(t) > 0:
            info["t_min"] = float(np.min(t))
            info["t_max"] = float(np.max(t))
            info["duration"] = float(t[-1] - t[0])

    return info


def print_length_summary(gestures, title, target_length=None):
    """
    Print useful sequence-length statistics for a gesture collection.
    """
    lengths = np.asarray(
        [get_sequence_length(g) for g in gestures],
        dtype=np.int64,
    )

    print(f"\n{title}")
    print("=" * len(title))

    print(f"Number of gestures: {len(lengths)}")

    if len(lengths) == 0:
        return

    print(f"Minimum length:     {lengths.min()}")
    print(f"Median length:      {np.median(lengths):.1f}")
    print(f"Mean length:        {lengths.mean():.2f}")
    print(f"95th percentile:    {np.percentile(lengths, 95):.1f}")
    print(f"99th percentile:    {np.percentile(lengths, 99):.1f}")
    print(f"Maximum length:     {lengths.max()}")

    if target_length is not None:
        shorter = np.sum(lengths < target_length)
        exact = np.sum(lengths == target_length)
        longer = np.sum(lengths > target_length)

        print(f"\nTarget length:      {target_length}")
        print(f"Shorter:            {shorter}")
        print(f"Exactly target:     {exact}")
        print(f"Longer:             {longer}")

        if longer > 0:
            offending_indices = np.where(
                lengths > target_length
            )[0]

            print(
                f"\nFirst {min(20, len(offending_indices))} "
                f"gestures longer than {target_length}:"
            )

            for idx in offending_indices[:20]:
                info = describe_gesture(gestures[idx])

                line = (
                    f"  index={idx:<6} "
                    f"length={info['length']:<5} "
                    f"shape={info['shape']}"
                )

                if "duration" in info:
                    line += (
                        f"  t=[{info['t_min']:.4f}, "
                        f"{info['t_max']:.4f}]"
                        f"  duration={info['duration']:.4f}s"
                    )

                print(line)

    # Show longest samples regardless of target.
    order = np.argsort(lengths)[::-1]

    print("\n10 longest gestures:")

    for idx in order[:10]:
        info = describe_gesture(gestures[idx])

        line = (
            f"  index={idx:<6} "
            f"length={info['length']:<5} "
            f"shape={info['shape']}"
        )

        if "duration" in info:
            line += (
                f"  t=[{info['t_min']:.4f}, "
                f"{info['t_max']:.4f}]"
                f"  duration={info['duration']:.4f}s"
            )

        print(line)


def install_preprocessing_debug_hook(dataset):
    """
    Temporarily intercept Dataset.pad_gestures().

    This lets us inspect the dataset at the exact point immediately before
    padding, without changing the shared dataset_loader implementation.
    """
    dataset_class = type(dataset)

    original_pad_gestures = dataset_class.pad_gestures

    def debug_pad_gestures(self, *args, **kwargs):
        if "num_points" in kwargs:
            num_points = kwargs["num_points"]
        elif len(args) >= 1:
            num_points = args[0]
        else:
            num_points = None

        print_length_summary(
            self.gestures,
            "State immediately BEFORE pad_gestures()",
            target_length=num_points,
        )

        print("\npad_gestures arguments")
        print("----------------------")
        print(f"args:   {args}")
        print(f"kwargs: {kwargs}")

        try:
            result = original_pad_gestures(
                self,
                *args,
                **kwargs,
            )

        except Exception as exc:
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("pad_gestures() FAILED")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"Exception: {type(exc).__name__}: {exc}")

            print(
                "\nIf 'Longer' above is greater than zero, "
                "pad_data() is being asked to handle sequences "
                "that already exceed num_points."
            )

            raise

        # Depending on implementation, result may be self or another Dataset.
        result_dataset = (
            result
            if hasattr(result, "gestures")
            else self
        )

        print_length_summary(
            result_dataset.gestures,
            "State immediately AFTER pad_gestures()",
            target_length=num_points,
        )

        return result

    dataset_class.pad_gestures = debug_pad_gestures

    return dataset_class, original_pad_gestures


# =============================================================================
# Split / labels
# =============================================================================

def prepare_data(dataset, seed=0, single_stroke=False):
    """
    Same deterministic train/validation/test split as torch-diffusion.
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
    Convert one-hot encoded gesture conditions to DeepGAN class labels.
    """
    if conditions.ndim != 2:
        raise ValueError(
            f"Expected 2D conditions, got {conditions.shape}"
        )

    return np.argmax(
        conditions,
        axis=1,
    )


# =============================================================================
# DeepGAN
# =============================================================================

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

    model.run_training_loop(
        data_split
    )

    model.save()


# =============================================================================
# Main
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Our wrapper arguments
    # -------------------------------------------------------------------------

    benchmark_args = parse_benchmark_arguments()

    print(
        f"Using benchmark config: "
        f"{benchmark_args.benchmark_config}"
    )

    print(
        f"Debug only: "
        f"{benchmark_args.debug_only}"
    )

    # -------------------------------------------------------------------------
    # Original DeepGAN arguments
    # -------------------------------------------------------------------------

    opt = Options()
    opt.parse()

    # -------------------------------------------------------------------------
    # Shared benchmark config
    # -------------------------------------------------------------------------

    with open(
        benchmark_args.benchmark_config,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    print("\nRelevant preprocessing configuration")
    print("------------------------------------")
    print(f"data_path:       {config['data_path']}")
    print(f"representation:  {config['representation']}")
    print(f"mode:            {config['mode']}")
    print(f"dt:              {config['dt']}")
    print(f"num_points:      {config['num_points']}")
    print(f"min_size:        {config['min_size']}")
    print(f"max_size:        {config['max_size']}")
    print(f"min_strokes:     {config['min_strokes']}")
    print(f"max_strokes:     {config['max_strokes']}")
    print(f"normalize:       {config['normalize']}")
    print(f"pad_value:       {config['pad_value']}")

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

    print(
        f"\nLoaded gestures before filtering: "
        f"{len(dataset)}"
    )

    print_length_summary(
        dataset.gestures,
        "Gesture lengths AFTER from_json()",
        target_length=config["num_points"],
    )

    # -------------------------------------------------------------------------
    # Same filtering as DDPM
    # -------------------------------------------------------------------------

    dataset = dataset.filter(
        config["class_filters"],
        config["condition_indices"],
    )

    print(
        f"\nLoaded gestures after filtering: "
        f"{len(dataset)}"
    )

    print_length_summary(
        dataset.gestures,
        "Gesture lengths AFTER filter()",
        target_length=config["num_points"],
    )

    # -------------------------------------------------------------------------
    # Install debug hook around pad_gestures()
    # -------------------------------------------------------------------------

    dataset_class, original_pad_gestures = (
        install_preprocessing_debug_hook(dataset)
    )

    # -------------------------------------------------------------------------
    # Run exactly the same preprocessing pipeline as DDPM
    # -------------------------------------------------------------------------

    print("\nRunning build_dataset_pipeline()")
    print("================================")

    try:
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

    finally:
        # Restore original method even if preprocessing fails.
        dataset_class.pad_gestures = original_pad_gestures

    # -------------------------------------------------------------------------
    # If we get here, preprocessing succeeded
    # -------------------------------------------------------------------------

    print("\nbuild_dataset_pipeline() completed successfully.")

    print_length_summary(
        dataset.gestures,
        "Gesture lengths AFTER complete preprocessing",
        target_length=config["num_points"],
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

    print("\nFinal split shapes")
    print("------------------")

    print(
        f"Train gestures: "
        f"{gesture_train.shape}"
    )

    print(
        f"Val gestures:   "
        f"{gesture_val.shape}"
    )

    print(
        f"Test gestures:  "
        f"{gesture_test.shape}"
    )

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
    # Fixed-dt checks
    # -------------------------------------------------------------------------

    if config["mode"] != "interpolate":
        raise RuntimeError(
            "This run is currently intended for the "
            "fixed-dt / interpolate experiment."
        )

    if gesture_train.ndim != 3:
        raise RuntimeError(
            f"Expected 3D training tensor, "
            f"got {gesture_train.shape}"
        )

    if gesture_train.shape[2] != 2:
        raise RuntimeError(
            "Fixed-dt data should contain exactly "
            f"(x, y), got {gesture_train.shape[2]} features."
        )

    if not np.isfinite(
        gesture_train
    ).all():
        raise RuntimeError(
            "Training data contains NaN or Inf."
        )

    # -------------------------------------------------------------------------
    # Conditions -> DeepGAN labels
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

    print("\nClasses")
    print("-------")
    print(
        f"Train: {np.unique(train_labels)}"
    )
    print(
        f"Val:   {np.unique(val_labels)}"
    )
    print(
        f"Test:  {np.unique(test_labels)}"
    )

    # -------------------------------------------------------------------------
    # DeepGAN sequence length
    # -------------------------------------------------------------------------

    sequence_length = gesture_train.shape[1]

    opt.resample_n = sequence_length

    print("\nDeepGAN sequence configuration")
    print("------------------------------")
    print(
        f"dt:             "
        f"{config['dt']} s"
    )
    print(
        f"sequence length:"
        f" {sequence_length}"
    )
    print(
        f"resample_n:     "
        f"{opt.resample_n}"
    )

    # -------------------------------------------------------------------------
    # Adapter
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
    # Original DeepGAN DataLoader
    # -------------------------------------------------------------------------

    data_split = deepgan_dataset.get_split()

    data_loader = (
        data_split.get_data_loader()
    )

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

    print(
        "\nData pipeline sanity check passed."
    )

    # -------------------------------------------------------------------------
    # Debug mode stops here
    # -------------------------------------------------------------------------

    if benchmark_args.debug_only:
        print(
            "\n--debug-only active: "
            "stopping before DeepGAN training."
        )
        return

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------

    if opt.use_cuda:

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )

        device = torch.device(
            "cuda:0"
        )

        print("\nUsing CUDA")
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        device = torch.device(
            "cpu"
        )

        print("\nUsing CPU")

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    print("\nStarting DeepGAN training")
    print("=========================")

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