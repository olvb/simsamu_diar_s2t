# helpers for speechbrain batching

from collections import Counter
import copy
import logging
from typing import List

import numpy as np
import speechbrain as sb
import torch

logger = logging.getLogger(__name__)


class WeightedDynamicBatchSampler(sb.dataio.sampler.DynamicBatchSampler):
    """
    Subclass of DynamicBatchSampler combining its features with ReproducibleWeightedRandomSampler
    """

    def __init__(
        self,
        dataset,
        max_batch_length: int,
        num_buckets: int = None,
        length_func=lambda x: x["duration"],
        shuffle: bool = True,
        batch_ordering: str = "random",
        max_batch_ex: int = None,
        bucket_boundaries: List[int] = [],
        lengths_list: List[int] = None,
        seed: int = 42,
        epoch: int = 0,
        drop_last: bool = False,
        verbose: bool = False,
        weighted_sampler_params: dict = None,
    ):
        if weighted_sampler_params is not None:
            self._weighted_sampler = (
                sb.dataio.sampler.ReproducibleWeightedRandomSampler(
                    **weighted_sampler_params
                )
            )
        else:
            self._weighted_sampler = None

        super().__init__(
            dataset=dataset,
            max_batch_length=max_batch_length,
            num_buckets=num_buckets,
            length_func=length_func,
            shuffle=shuffle,
            batch_ordering=batch_ordering,
            max_batch_ex=max_batch_ex,
            bucket_boundaries=bucket_boundaries,
            lengths_list=lengths_list,
            seed=seed,
            epoch=epoch,
            drop_last=drop_last,
            verbose=verbose,
        )

    def _generate_batches(self):
        ### Copy/paste of DynamicBatchSampler._generate_batches
        ### with injection of self._weighted_sampler

        logger.info("DynamicBatchSampler: Generating dynamic batches")
        ### begining of modified code
        if self._weighted_sampler is not None:
            sampler = self._weighted_sampler
        elif self._shuffle_ex:
            ### end of modified code
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch)
            sampler = torch.randperm(len(self._dataset), generator=g).tolist()  # type: ignore
        else:
            # take examples as they are: e.g. they have been sorted
            sampler = range(len(self._dataset))  # type: ignore

        self._batches = []
        bucket_batches = [[] for i in self._bucket_lens]

        stats_tracker = [
            {"min": np.inf, "max": -np.inf, "tot": 0, "n_ex": 0}
            for i in self._bucket_lens
        ]

        for idx in sampler:
            # length of pre-sampled audio
            item_len = self._ex_lengths[str(idx)]
            # bucket to fill up most padding
            bucket_id = np.searchsorted(self._bucket_boundaries, item_len)
            # fill audio's duration into that bucket
            bucket_batches[bucket_id].append(idx)

            stats_tracker[bucket_id]["min"] = min(
                stats_tracker[bucket_id]["min"], item_len
            )
            stats_tracker[bucket_id]["max"] = max(
                stats_tracker[bucket_id]["max"], item_len
            )
            stats_tracker[bucket_id]["tot"] += item_len
            stats_tracker[bucket_id]["n_ex"] += 1
            # track #samples - why not duration/#frames; rounded up?
            # keep track of durations, if necessary

            if (
                len(bucket_batches[bucket_id]) >= self._bucket_lens[bucket_id]
                or len(bucket_batches[bucket_id]) >= self._max_batch_ex
            ):
                self._batches.append(bucket_batches[bucket_id])
                bucket_batches[bucket_id] = []
                # keep track of durations

        # Dump remaining batches
        if not self._drop_last:
            for batch in bucket_batches:
                if batch:
                    self._batches.append(batch)

        self._permute_batches()  # possibly reorder batches

        if self._epoch == 0:  # only log at first epoch
            # frames per batch & their padding remaining
            boundaries = [0] + self._bucket_boundaries.tolist()

            for bucket_indx in range(len(self._bucket_boundaries)):
                try:
                    num_batches = stats_tracker[bucket_indx]["tot"] // (
                        self._max_batch_length
                    )
                    pad_factor = (
                        stats_tracker[bucket_indx]["max"]
                        - stats_tracker[bucket_indx]["min"]
                    ) / (
                        stats_tracker[bucket_indx]["tot"]
                        / stats_tracker[bucket_indx]["n_ex"]
                    )
                except ZeroDivisionError:
                    num_batches = 0
                    pad_factor = 0

                logger.debug(
                    (
                        "DynamicBatchSampler: Bucket {} with boundary {:.1f}-{:.1f} and "
                        + "batch_size {}: Num Examples {:.1f}, Num Full Batches {:.3f}, Pad Factor {:.3f}."
                    ).format(
                        bucket_indx,
                        boundaries[bucket_indx],
                        boundaries[bucket_indx + 1],
                        self._bucket_lens[bucket_indx],
                        stats_tracker[bucket_indx]["n_ex"],
                        num_batches,
                        pad_factor * 100,
                    )
                )

            if self.verbose:
                batch_stats = {
                    "tot_frames": [],
                    "tot_pad_frames": [],
                    "pad_%": [],
                }
                for batch in self._batches:
                    tot_frames = sum([self._ex_lengths[str(idx)] for idx in batch])
                    batch_stats["tot_frames"].append(tot_frames)
                    max_frames = max([self._ex_lengths[str(idx)] for idx in batch])
                    tot_pad = sum(
                        [max_frames - self._ex_lengths[str(idx)] for idx in batch]
                    )
                    batch_stats["tot_pad_frames"].append(tot_pad)
                    batch_stats["pad_%"].append(tot_pad / tot_frames * 100)

                padding_details = "Batch {} with {:.1f} frames with {} files - {:.1f} padding, {:.2f} (%) of total."
                padding_details = "DynamicBatchSampler: " + padding_details
                for i in range(len(self._batches)):
                    logger.debug(
                        padding_details.format(
                            i,
                            batch_stats["tot_frames"][i],
                            len(self._batches[i]),
                            batch_stats["tot_pad_frames"][i],
                            batch_stats["pad_%"][i],
                        )
                    )

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self._weighted_sampler.set_epoch(epoch)


def init_batch_sampler(
    hparams, dataset, epoch=0, batch_ordering=None, subset_by_source=None
):
    """
    Init speechbrain dynamic batch samplers that will try to pack as many samples
    as a batch based on the max_batch_length and nb_batch_buckets hparams.

    The goal is to handle efficiently in terms of GPU-memory batches of
    sequences of varying length. Refer to the speechbrain documentation of
    DynamicBatchSampler for more details.

    NB: We use a custom subclass of DynamicBatchSampler to mix the
    features of DynamicBatchSampler and ReproducibleWeightedRandomSampler
    """

    if batch_ordering is None:
        batch_ordering = hparams["batch_ordering"]

    if subset_by_source:
        weighted_sampler_params = compute_weighted_sampler_params(
            dataset, subset_by_source
        )
    else:
        weighted_sampler_params = None

    return WeightedDynamicBatchSampler(
        dataset,
        max_batch_length=hparams["max_batch_length"],
        num_buckets=hparams["nb_batch_buckets"],
        shuffle=False,
        batch_ordering=batch_ordering,
        length_func=lambda x: x["duration"],
        epoch=epoch,
        verbose=False,
        drop_last=False,
        weighted_sampler_params=weighted_sampler_params,
    )


def compute_weighted_sampler_params(dataset, subset_by_source):
    """
    Compute appropriate params for
    speechbrain.dataio.sampler.ReproducibleWeightedRandomSampler to honor values
    in subset_by_source.

    For instance, if we a dataset with 10000 samples having "cv" as source and
    1000 samples having "eslo2" a source, and if subset_by_source == {"cv": 0.1,
    "eslo2": 1.0}, we will compute the weights and num_samples params to make
    sure that at each epoch, 10000*0.1 == 1000 "cv" samples are used and
    1000*1.0 == 1000 "eslo2" samples are used.
    """

    # make a copy of the dataset without the wav output key
    # to avoid reading all audio files when iterating over
    dataset_copy = copy.copy(dataset)
    dataset_copy.pipeline = copy.deepcopy(dataset_copy.pipeline)
    dataset_copy.set_output_keys(["source"])

    sample_sources = [sample["source"] for sample in dataset_copy]
    nb_samples_total = len(sample_sources)

    nb_by_source = Counter(sample_sources)
    ratio_by_source = {
        source: nb_by_source[source] / nb_samples_total for source in subset_by_source
    }
    weights_by_source = {
        source: subset_by_source[source] / ratio_by_source[source]
        for source in subset_by_source
    }
    sample_weights = [
        weights_by_source[sample_source] for sample_source in sample_sources
    ]
    nb_samples = round(
        sum(
            nb_by_source[source] * subset_by_source[source]
            for source in subset_by_source
        )
    )

    return dict(
        weights=sample_weights,
        num_samples=nb_samples,
        replacement=False,
    )
