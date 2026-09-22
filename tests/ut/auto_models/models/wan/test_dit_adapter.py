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
"""Unit tests for the Wan DiT provider's Trainer-facing config contract."""

import unittest
from unittest.mock import patch

import torch
from diffusers.configuration_utils import ConfigMixin

from hyper_parallel.models.wan.dit_adapter import WanDiTModelProvider


class _FakeWanTransformer(torch.nn.Module, ConfigMixin):
    config_name = "config.json"

    def __init__(self, **config):
        super().__init__()
        self.register_to_config(**config)
        self.register_buffer("derived_buffer", torch.tensor([1.0, 2.0]), persistent=False)


class TestWanDiTModelProvider(unittest.TestCase):
    """Wan-local adaptation must satisfy BaseTrainer without changing it."""

    def test_build_model_exposes_transformers_style_config_dict(self):
        with patch.object(
            WanDiTModelProvider,
            "model_class",
            _FakeWanTransformer,
        ):
            model = WanDiTModelProvider.build_model({"dim": 128, "num_layers": 2})

        self.assertEqual(model.config.to_dict(), {"dim": 128, "num_layers": 2})
        self.assertEqual(dict(model.config), {"dim": 128, "num_layers": 2})

    def test_build_model_finalizes_preserved_nonpersistent_buffers(self):
        with patch.object(
            WanDiTModelProvider,
            "model_class",
            _FakeWanTransformer,
        ):
            model = WanDiTModelProvider.build_model({"dim": 128})

        expected = model.derived_buffer.clone()
        setattr(model.derived_buffer, "_is_hf_initialized", False)
        setattr(model, "_is_hf_initialized", False)
        model.initialize_weights()

        self.assertTrue(torch.equal(model.derived_buffer, expected))
        self.assertTrue(getattr(model.derived_buffer, "_is_hf_initialized", False))
        self.assertTrue(getattr(model, "_is_hf_initialized", False))


if __name__ == "__main__":
    unittest.main()
