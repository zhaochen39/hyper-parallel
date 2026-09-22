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
"""Unit tests for generic Diffusers DiT model construction."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from diffusers.configuration_utils import ConfigMixin

from hyper_parallel.models._diffusers import (
    DiffusersDiTModelProvider,
    build_diffusers_model,
    normalize_diffusers_config,
    resolve_diffusers_component_path,
)


class _FakeDiffusersModel(torch.nn.Module, ConfigMixin):
    config_name = "config.json"

    def __init__(self, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.register_to_config(hidden_size=hidden_size, num_layers=num_layers)
        self.register_buffer("derived_buffer", torch.tensor([1.0, 2.0]), persistent=False)

    def forward(self, hidden_states, timestep=None):
        del timestep
        return hidden_states


class _FakeProvider(DiffusersDiTModelProvider):
    model_class = _FakeDiffusersModel


class TestDiffusersDiTModelProvider(unittest.TestCase):
    """Generic provider behavior shared by upstream Diffusers DiT families."""

    def test_normalize_config_filters_diffusers_metadata(self):
        """Only constructor arguments survive generic config normalization."""
        config = normalize_diffusers_config(
            _FakeDiffusersModel,
            {
                "hidden_size": 128,
                "num_layers": 2,
                "_class_name": "FakeDiffusersModel",
                "unused": True,
            },
        )

        self.assertEqual(config, {"hidden_size": 128, "num_layers": 2})

    def test_build_model_installs_shared_loading_compatibility(self):
        """Pretrained builds expose config and deferred-loading compatibility."""
        model = build_diffusers_model(
            _FakeDiffusersModel,
            {"hidden_size": 128},
            for_pretrained_loading=True,
        )
        expected = model.derived_buffer.clone()
        setattr(model.derived_buffer, "_is_hf_initialized", False)

        model.initialize_weights()

        self.assertEqual(model.config.to_dict(), {"hidden_size": 128, "num_layers": 1})
        self.assertTrue(torch.equal(model.derived_buffer, expected))
        self.assertTrue(getattr(model.derived_buffer, "_is_hf_initialized", False))

    def test_resolve_component_path_accepts_pipeline_root_and_component(self):
        """Both a pipeline root and direct component directory resolve locally."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            component = root / "transformer"
            component.mkdir()
            (component / "config.json").write_text("{}", encoding="utf-8")

            from_root = resolve_diffusers_component_path(str(root), "transformer")
            from_component = resolve_diffusers_component_path(str(component), "transformer")

        self.assertEqual(from_root, str(component))
        self.assertEqual(from_component, str(component))

    def test_provider_loads_json_and_returns_checkpoint_path(self):
        """The default provider loads standard JSON without a family override."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            component = Path(temporary_directory) / "transformer"
            component.mkdir()
            (component / "config.json").write_text(
                json.dumps({"hidden_size": 128, "num_layers": 2, "unused": True}),
                encoding="utf-8",
            )

            config, checkpoint_path = _FakeProvider.resolve_pretrained(
                str(component),
                transformer_subfolder="transformer",
                config_path=None,
            )

        self.assertEqual(config, {"hidden_size": 128, "num_layers": 2})
        self.assertEqual(checkpoint_path, str(component))


if __name__ == "__main__":
    unittest.main()
