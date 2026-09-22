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
"""Unit tests for generic DiT forward input routing."""

import inspect
import unittest

import torch

from hyper_parallel.trainer.dit_trainer import _select_forward_inputs


class _ExplicitForwardModel(torch.nn.Module):
    def forward(self, latent_input, noise_level, optional_context=None):
        del noise_level, optional_context
        return latent_input


class _WrappedForwardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._hp_forward_signature = inspect.signature(_ExplicitForwardModel.forward)

    def forward(self, **kwargs):
        return kwargs


class _KwargsForwardModel(torch.nn.Module):
    def forward(self, **kwargs):
        return kwargs


class TestDiTForwardInputRouting(unittest.TestCase):
    """Model inputs are selected from forward signatures instead of field lists."""

    def test_selects_explicit_inputs_and_keeps_loss_metadata_out(self):
        """Explicit forward names select model data without naming loss fields."""
        latent_input = torch.ones(1)
        noise_level = torch.zeros(1)

        model_inputs = _select_forward_inputs(
            _ExplicitForwardModel(),
            {
                "latent_input": [latent_input],
                "noise_level": [noise_level],
                "loss_target": [torch.randn(1)],
            },
        )

        self.assertEqual(set(model_inputs), {"latent_input", "noise_level"})
        self.assertIs(model_inputs["latent_input"], latent_input)
        self.assertIs(model_inputs["noise_level"], noise_level)

    def test_uses_preserved_upstream_signature_after_wrapping(self):
        """Preserved signatures keep routing stable after wrappers obscure forward."""
        model_inputs = _select_forward_inputs(
            _WrappedForwardModel(),
            {
                "latent_input": [torch.ones(1)],
                "noise_level": [torch.zeros(1)],
                "loss_target": [torch.randn(1)],
            },
        )

        self.assertEqual(set(model_inputs), {"latent_input", "noise_level"})

    def test_kwargs_forward_receives_complete_conditioned_batch(self):
        """Models with open keyword inputs retain all conditioned fields."""
        model_inputs = _select_forward_inputs(
            _KwargsForwardModel(),
            {"hidden_states": [torch.ones(1)], "training_target": [torch.zeros(1)]},
        )

        self.assertEqual(set(model_inputs), {"hidden_states", "training_target"})

    def test_preserves_tuple_model_inputs(self):
        """Semantic tuples are not mistaken for collator-preserved sample lists."""
        model_inputs = _select_forward_inputs(
            _ExplicitForwardModel(),
            {"latent_input": (torch.ones(1), torch.zeros(1)), "noise_level": [torch.zeros(1)]},
        )

        self.assertIsInstance(model_inputs["latent_input"], tuple)


if __name__ == "__main__":
    unittest.main()
