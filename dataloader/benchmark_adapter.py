import copy

import numpy as np
import torch

from dataloader.dataset import Dataset, DataSplit
from dataloader.sample import Sample


class BenchmarkSample(Sample):
    """
    Sample variant for already-preprocessed benchmark data.

    Unlike the original DeepGAN Sample, this does NOT perform another
    spatial resampling step in to_torch().
    """

    def to_torch(self, resample_n):
        result = copy.deepcopy(self)

        # Data has already been resampled/interpolated and padded
        # by the shared dataset pipeline.
        # resample_n is therefore intentionally ignored.
        result.x = torch.from_numpy(
            np.asarray(self.x, dtype=np.float32)
        )

        if self.y is None:
            raise RuntimeError(
                f"BenchmarkSample has no class index assigned "
                f"(label={self.label})."
            )

        result.y = torch.tensor(
            [self.y],
            dtype=torch.int64,
        )

        return result


class IdentityNormalizer:
    """
    No-op normalizer.

    The benchmark data is already normalized/preprocessed by the shared
    dataset_loader pipeline, so DeepGAN must not normalize it again.
    """

    def __init__(self):
        self.factors_computed = True

    def compute_factors(self, train_set):
        pass

    def normalize_list(self, sample_list):
        pass

    def normalize(self, sample):
        return sample

    def unnormalize_list(self, samples):
        if isinstance(samples, torch.Tensor):
            return samples.detach().cpu().numpy()

        return np.asarray(samples)

    def unnormalize(self, sample):
        if isinstance(sample, torch.Tensor):
            return sample.detach().cpu().numpy()

        return sample


class BenchmarkDataset(Dataset):
    """
    Adapter between the shared preprocessing pipeline and DeepGAN.

    Expects gestures that are already:
        - filtered
        - resampled/interpolated
        - padded if necessary
        - normalized
        - split into train/validation/test

    Only the training subset should be passed to this class.
    """

    def __init__(self, opt, gestures, labels):
        super().__init__(opt)

        if len(gestures) != len(labels):
            raise ValueError(
                f"Number of gestures ({len(gestures)}) does not match "
                f"number of labels ({len(labels)})."
            )

        if len(gestures) == 0:
            raise ValueError(
                "Cannot construct BenchmarkDataset from empty data."
            )

        # Convert labels to plain Python ints.
        #
        # This avoids numpy scalar types leaking into the original
        # DeepGAN class mappings.
        labels = [
            int(label)
            for label in labels
        ]

        samples = [
            BenchmarkSample(
                x=gesture,
                label=label,
            )
            for gesture, label in zip(gestures, labels)
        ]

        # Explicitly define a stable class order.
        #
        # Example:
        #   labels = [3, 0, 2, 1, ...]
        #
        # becomes:
        #   class_to_idx = {
        #       0: 0,
        #       1: 1,
        #       2: 2,
        #       3: 3,
        #       4: 4,
        #   }
        dataset_classes = sorted(
            set(labels)
        )

        self._fill(
            samples,
            dataset_classes=dataset_classes,
        )

        # Important:
        #
        # The original Dataset._fill() only assigns sample.y automatically
        # when dataset_classes is None.
        #
        # Because we explicitly provide dataset_classes to enforce stable
        # class ordering, we must assign y ourselves.
        for sample in self.samples:
            sample.y = self.class_to_idx[
                sample.label
            ]

        # Sanity check.
        for sample in self.samples:
            if sample.y is None:
                raise RuntimeError(
                    f"Failed to assign DeepGAN class index "
                    f"for label {sample.label}."
                )

        # We do not use the original DeepGAN visualizers for the benchmark.
        #
        # This is especially important for future (x, y, t) experiments:
        # the third dimension represents time, not a spatial z coordinate.
        self._visualizer = None

    def get_split(self, normalizer=None):
        """
        Return the original DeepGAN DataSplit while disabling DeepGAN's
        additional MinMax normalization.

        The shared dataset pipeline has already performed normalization.
        """
        return DataSplit(
            self,
            normalizer=IdentityNormalizer(),
        )