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

        # Data has already been resampled by our shared dataset pipeline.
        # resample_n is therefore intentionally ignored.
        result.x = torch.from_numpy(
            np.asarray(self.x, dtype=np.float32)
        )

        result.y = torch.ones(
            (1,),
            dtype=torch.int64
        ) * self.y

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
    Adapter between our shared preprocessing pipeline and DeepGAN.

    Expects gestures that are already:
        - filtered
        - resampled/interpolated
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
            raise ValueError("Cannot construct BenchmarkDataset from empty data.")

        samples = [
            BenchmarkSample(
                x=gesture,
                label=label,
            )
            for gesture, label in zip(gestures, labels)
        ]

        # Let the original DeepGAN Dataset handle:
        # - samples
        # - class_to_idx
        # - idx_to_class
        # - y indices
        # - num_classes
        # - num_features
        self._fill(
    samples,
    dataset_classes=list(
        range(len(np.unique(labels)))
    ),
)

        # We do not use the original DeepGAN visualizers for the benchmark.
        # In particular, (x, y, t) must not be interpreted as spatial 3D data.
        self._visualizer = None

    def get_split(self, normalizer=None):
        """
        Return the original DeepGAN DataSplit, but disable its additional
        MinMax normalization.
        """
        return DataSplit(
            self,
            normalizer=IdentityNormalizer(),
        )