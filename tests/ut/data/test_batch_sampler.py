# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Unit tests for distributed Dataset batch sampling."""

import unittest

from torch.utils.data import DistributedSampler

from hyper_parallel.data.parallel.batch_sampler import build_dataset_batch_sampler


class _IndexDataset:
    """Minimal mapping Dataset used to compute PyTorch reference indices."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


class TestDistributedDatasetBatchSampler(unittest.TestCase):
    """The HyperParallel sampler must match PyTorch DistributedSampler."""

    def test_matches_reference_without_shuffle(self) -> None:
        """Drop-last and padding behavior match the reference sampler."""
        for drop_last in (False, True):
            for rank in range(2):
                with self.subTest(drop_last=drop_last, rank=rank):
                    reference = DistributedSampler(
                        _IndexDataset(11),
                        num_replicas=2,
                        rank=rank,
                        shuffle=False,
                        drop_last=drop_last,
                    )
                    sampler = build_dataset_batch_sampler(
                        total_samples=11,
                        micro_batch_size=2,
                        global_batch_size=4,
                        dp_rank=rank,
                        dp_world_size=2,
                        drop_last=drop_last,
                        sampler_type="distributed",
                        shuffle=False,
                    )

                    actual = [index for batch in sampler for index in batch]
                    expected = list(reference)
                    if drop_last:
                        expected = expected[:len(expected) - len(expected) % 2]

                    self.assertEqual(
                        actual,
                        expected,
                        msg=f"drop_last={drop_last}, rank={rank}, expected={expected}, actual={actual}",
                    )

    def test_matches_reference_shuffle_for_each_epoch(self) -> None:
        """Epoch-dependent shuffling uses the same seed contract as PyTorch."""
        dataset = _IndexDataset(12)
        reference = DistributedSampler(
            dataset,
            num_replicas=2,
            rank=1,
            shuffle=True,
            seed=17,
            drop_last=True,
        )
        sampler = build_dataset_batch_sampler(
            total_samples=len(dataset),
            micro_batch_size=2,
            global_batch_size=4,
            dp_rank=1,
            dp_world_size=2,
            drop_last=True,
            sampler_type="distributed",
            shuffle=True,
            seed=17,
        )

        for epoch in (0, 3):
            with self.subTest(epoch=epoch):
                reference.set_epoch(epoch)
                sampler.set_epoch(epoch)
                actual = [index for batch in sampler for index in batch]
                expected = list(reference)

                self.assertEqual(
                    actual,
                    expected,
                    msg=f"epoch={epoch}, expected={expected}, actual={actual}",
                )


if __name__ == "__main__":
    unittest.main()
