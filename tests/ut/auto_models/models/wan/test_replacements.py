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
"""Unit tests for the official Diffusers Wan attention backend adapter."""

import unittest

from torch import nn

from hyper_parallel.models.wan.adapter.replacements import (
    replace_wan_attention_backend,
)
from tests.common.mark_utils import arg_mark


class _FakeWanAttention(nn.Module):
    """Minimal stand-in for Diffusers ``WanAttention``."""

    def __init__(self) -> None:
        super().__init__()
        self.backend = None

    def set_attention_backend(self, backend: str) -> None:
        self.backend = backend

    def forward(self, hidden_states):
        return hidden_states


class TestWanAttentionBackendReplacement(unittest.TestCase):
    """The adapter must configure, not reimplement, official attention."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_configures_backend_and_preserves_module_identity(self):
        """The replacement keeps the official module and all of its state."""
        module = _FakeWanAttention()

        replacement = replace_wan_attention_backend(
            module=module,
            module_fqn="blocks.0.attn1",
            context={},
            backend="_native_npu",
        )

        self.assertIs(replacement, module)
        self.assertEqual(module.backend, "_native_npu")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_rejects_module_without_diffusers_backend_contract(self):
        """A mismatched Diffusers release fails before training starts."""
        with self.assertRaisesRegex(TypeError, "set_attention_backend"):
            replace_wan_attention_backend(
                module=nn.Identity(),
                module_fqn="blocks.0.attn1",
                context={},
                backend="_native_npu",
            )


if __name__ == "__main__":
    unittest.main()
