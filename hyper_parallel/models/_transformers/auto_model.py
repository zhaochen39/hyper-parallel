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
"""HyperAutoModel classes — HF-compatible model interface.

Following design doc 01_hf_compatibility_layer.md §6.
Stub — provides from_pretrained/from_config as entry points.
"""

import importlib
import logging
from typing import Any, Literal, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    PretrainedConfig,
    PreTrainedModel,
)

from hyper_parallel.models._transformers.model_builder import (
    _init_model,
    apply_model_infrastructure,
    instantiate_infrastructure,
)
from hyper_parallel.models._transformers.config_resolver import get_hf_config, get_is_hf_model
from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models.build_options import get_device_id, get_device_type  # pylint: disable=syntax-error
from hyper_parallel.models.build_options import CompileConfig
from hyper_parallel.models.registry import get_model_adapter

logger = logging.getLogger(__name__)


def _current_device() -> torch.device:
    """Return the current accelerator device, or CPU when no accelerator exists."""
    device_type = get_device_type()
    if device_type == "cpu":
        return torch.device("cpu")
    return torch.device(device_type, get_device_id())


class _BaseHyperAutoModelClass:
    """Shared from_pretrained / from_config logic.

    Following design doc 01 §6.1-6.2.
    """

    @classmethod
    def _from_pretrained_parent_class(cls, *args, **kwargs):
        """Delegate to the parent HuggingFace AutoModel class.

        Used by the HF-native path in _init_model so that the model is loaded
        through the standard transformers checkpoint logic.
        """
        # Mixin: `super()` resolves to the HF AutoModel base at runtime (see
        # HyperAutoModelForCausalLM etc. below), which provides from_pretrained.
        return super().from_pretrained(*args, **kwargs)  # pylint: disable=E1101

    @classmethod
    def _from_config_parent_class(cls, *args, **kwargs):
        """Delegate to the parent Hugging Face AutoModel ``from_config`` path."""
        return super().from_config(*args, **kwargs)  # pylint: disable=no-member

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        distributed_setup: Optional[DistributedSetup] = None,
        backend: Optional[Any] = None,
        peft_config: Optional[Any] = None,
        torch_dtype: Union[str, torch.dtype] = "auto",
        attn_implementation: str = "sdpa",
        force_hf: bool = False,
        validate_placement: bool = False,
        qat_config: Optional[Any] = None,
        fp8_config: Optional[Any] = None,
        compile_config: Optional[Union[CompileConfig, dict]] = None,
        freeze_config: Optional[Any] = None,
        activation_checkpoint: Optional[str] = None,
        swap_inputs: bool = False,
        activation_swap: str = "none",
        model_init_dtype: Optional[Literal["float16", "bfloat16", "float32"]] = None,
        **kwargs: Any,
    ) -> PreTrainedModel:
        """HF-compatible from_pretrained entry point.

        Following design doc 01 §6.2:
        ① resolve distributed_setup → MeshContext
        ② instantiate_infrastructure → ShardingPlanner + FSDP2Manager
        ③ AutoConfig.from_pretrained → hf_config
        ④ get_is_hf_model → custom/HF path
        ⑤ _build_model → meta + shard + load
        """
        if distributed_setup is None:
            distributed_setup = DistributedSetup()
        mesh = distributed_setup.mesh_context

        # ② Instantiate infrastructure
        sharding_planner, fsdp2_manager = instantiate_infrastructure(
            distributed_setup=distributed_setup,
            device=_current_device(),
        )

        # ③ Get HF config
        hf_config = get_hf_config(
            pretrained_model_name_or_path, attn_implementation, torch_dtype, **kwargs
        )

        # ④ Determine model path
        is_hf_model = get_is_hf_model(hf_config, force_hf)

        # ⑤ Build model
        return cls._build_model(
            pretrained_model_name_or_path,
            *model_args,
            is_hf_model=is_hf_model,
            hf_config=hf_config,
            mesh=mesh,
            sharding_planner=sharding_planner,
            fsdp2_manager=fsdp2_manager,
            backend=backend,
            peft_config=peft_config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            validate_placement=validate_placement,
            load_base_model=True,
            distributed_setup=distributed_setup,
            qat_config=qat_config,
            fp8_config=fp8_config,
            compile_config=compile_config,
            freeze_config=freeze_config,
            activation_checkpoint=activation_checkpoint,
            swap_inputs=swap_inputs,
            activation_swap=activation_swap,
            model_init_dtype=model_init_dtype,
            **kwargs,
        )

    @classmethod
    def from_config(  # pylint: disable=unused-argument
        cls,
        config: PretrainedConfig,
        *model_args: Any,
        distributed_setup: Optional[DistributedSetup] = None,
        device_mesh: Optional[Any] = None,
        backend: Optional[Any] = None,
        peft_config: Optional[Any] = None,
        torch_dtype: Union[str, torch.dtype] = "auto",
        attn_implementation: str = "sdpa",
        validate_placement: bool = False,
        qat_config: Optional[Any] = None,
        fp8_config: Optional[Any] = None,
        compile_config: Optional[Union[CompileConfig, dict]] = None,
        freeze_config: Optional[Any] = None,
        activation_checkpoint: Optional[str] = None,
        swap_inputs: bool = False,
        activation_swap: str = "none",
        model_init_dtype: Optional[Literal["float16", "bfloat16", "float32"]] = None,
        **kwargs: Any,
    ) -> PreTrainedModel:
        """Build model from PretrainedConfig (no weight loading).

        Following design doc 01 §6.1.
        """
        if distributed_setup is None:
            distributed_setup = DistributedSetup()
        mesh = distributed_setup.mesh_context

        sharding_planner, fsdp2_manager = instantiate_infrastructure(
            distributed_setup=distributed_setup,
            device=_current_device(),
        )

        is_hf_model = get_is_hf_model(config, force_hf=False)

        return cls._build_model(
            None,
            *model_args,
            is_hf_model=is_hf_model,
            hf_config=config,
            mesh=mesh,
            sharding_planner=sharding_planner,
            fsdp2_manager=fsdp2_manager,
            backend=backend,
            peft_config=peft_config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            validate_placement=validate_placement,
            load_base_model=False,
            distributed_setup=distributed_setup,
            qat_config=qat_config,
            fp8_config=fp8_config,
            compile_config=compile_config,
            freeze_config=freeze_config,
            activation_checkpoint=activation_checkpoint,
            swap_inputs=swap_inputs,
            activation_swap=activation_swap,
            model_init_dtype=model_init_dtype,
            **kwargs,
        )

    @classmethod
    def _build_model(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        is_hf_model,
        hf_config,
        mesh,
        sharding_planner,
        fsdp2_manager,
        backend,
        peft_config,
        torch_dtype,
        attn_implementation,
        validate_placement,
        load_base_model,
        distributed_setup=None,
        qat_config=None,
        fp8_config=None,
        compile_config=None,
        freeze_config=None,
        activation_checkpoint: Optional[str] = None,
        swap_inputs: bool = False,
        activation_swap: str = "none",
        model_init_dtype: Optional[Literal["float16", "bfloat16", "float32"]] = None,
        **kwargs,
    ) -> PreTrainedModel:
        """Core model building orchestration.

        Following design doc 01 §6.3:
        Step 1: Determine meta device
        Step 2: Build model (meta or real device)
        Step 3-12: apply_model_infrastructure (PEFT, QAT, ShardingPlan,
        activation checkpoint, FSDP2, load, layer compile)
        """
        # Lazy imports: no_init_weights moved between transformers submodules
        # across versions (hence the ImportError fallback).
        # pylint: disable=import-outside-toplevel
        from contextlib import nullcontext
        from transformers.modeling_utils import ContextManagers
        try:
            from transformers.modeling_utils import no_init_weights
        except ImportError:
            from transformers.initialization import no_init_weights
        from hyper_parallel import init_empty_weights
        # pylint: enable=import-outside-toplevel

        # Step 1: Determine meta device (inline world-size probe: auto_models
        # must not import the trainer runtime).
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        is_meta_device = (
            world_size > 1 or not is_hf_model
        ) and kwargs.get("quantization_config") is None

        init_ctx = (
            ContextManagers([no_init_weights(), init_empty_weights()])
            if is_meta_device
            else nullcontext()
        )

        # Step 2: Build model
        with init_ctx:
            _, model = _init_model(
                cls,
                pretrained_model_name_or_path,
                hf_config,
                attn_implementation,
                torch_dtype,
                is_hf_model,
                *model_args,
                backend=backend,
                **kwargs,
            )

        # Step 3-12: Apply infrastructure
        model = apply_model_infrastructure(
            model,
            mesh=mesh,
            sharding_planner=sharding_planner,
            fsdp2_manager=fsdp2_manager,
            peft_config=peft_config,
            qat_config=qat_config,
            fp8_config=fp8_config,
            freeze_config=freeze_config,
            compile_config=compile_config,
            is_meta_device=is_meta_device,
            is_hf_model=is_hf_model,
            device=_current_device(),
            load_base_model=load_base_model,
            pretrained_path=pretrained_model_name_or_path,
            validate_placement=validate_placement,
            distributed_setup=distributed_setup,
            activation_checkpoint=activation_checkpoint,
            swap_inputs=swap_inputs,
            activation_swap=activation_swap,
            model_init_dtype=model_init_dtype,
        )

        model.train()
        return model


class HyperAutoModelForCausalLM(_BaseHyperAutoModelClass, AutoModelForCausalLM):
    """Hyper-Parallel CausalLM — equivalent to AutoModelForCausalLM.

    Following design doc 01 §6.1.
    """


class HyperAutoModelForImageTextToText(_BaseHyperAutoModelClass, AutoModelForImageTextToText):
    """Hyper-Parallel VLM — equivalent to AutoModelForImageTextToText."""


class HyperAutoModelForSequenceClassification(_BaseHyperAutoModelClass, AutoModelForSequenceClassification):
    """Hyper-Parallel SequenceClassification."""


def _resolve_dit_provider(provider: Any = None, model_type: Optional[str] = None) -> Any:
    """Resolve a DiT provider from an explicit path or model-family spec."""
    if provider is None:
        if not model_type:
            raise ValueError("DiT AutoModel requires model_type or model_provider")
        adapter_spec = get_model_adapter(model_type)
        if adapter_spec is None or adapter_spec.dit is None:
            raise ValueError(f"No DiT provider registered for model_type={model_type!r}")
        return adapter_spec.dit()
    if isinstance(provider, str):
        parts = provider.split(".")
        for split_at in range(len(parts), 0, -1):
            try:
                value = importlib.import_module(".".join(parts[:split_at]))
            except ModuleNotFoundError as exc:
                if exc.name == ".".join(parts[:split_at]) or ".".join(parts[:split_at]).startswith(f"{exc.name}."):
                    continue
                raise
            for attribute in parts[split_at:]:
                value = getattr(value, attribute)
            return value
        raise ImportError(f"Unable to import DiT model provider {provider!r}")
    return provider


resolve_dit_provider = _resolve_dit_provider


class HyperAutoModelForDiT(_BaseHyperAutoModelClass):
    """Generic AutoModel facade for Diffusers-style DiT families.

    A family-specific provider owns config/checkpoint format and model class;
    this facade owns only the shared HyperParallel build lifecycle.
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[str] = None,
        *,
        model_provider: Any = None,
        model_type: Optional[str] = None,
        distributed_setup: Optional[DistributedSetup] = None,
        peft_config: Optional[Any] = None,
        torch_dtype: Union[str, torch.dtype] = "bfloat16",
        attn_implementation: str = "sdpa",
        activation_checkpoint: Optional[str] = None,
        swap_inputs: bool = False,
        activation_swap: str = "none",
        compile_config: Optional[Union[CompileConfig, dict]] = None,
        model_init_dtype: Optional[Literal["float16", "bfloat16", "float32"]] = None,
        validate_placement: bool = False,
        transformer_subfolder: Optional[str] = "transformer",
        config_path: Optional[str] = None,
        task: str = "t2v",
        **kwargs: Any,
    ) -> torch.nn.Module:
        """Resolve a provider and run the common DiT construction pipeline."""
        provider = _resolve_dit_provider(model_provider, model_type)
        if distributed_setup is None:
            raise ValueError("DiT AutoModel requires distributed_setup from BaseTrainer")
        prepare = getattr(provider, "prepare", None)
        if callable(prepare):
            prepare(distributed_setup, peft_config=peft_config, **kwargs)
        config, checkpoint_path = provider.resolve_pretrained(
            pretrained_model_name_or_path,
            transformer_subfolder=transformer_subfolder,
            config_path=config_path,
            task=task,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            **kwargs,
        )
        planner, fsdp = instantiate_infrastructure(
            distributed_setup=distributed_setup, device=_current_device()
        )
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        is_meta_device = checkpoint_path is not None or world_size > 1
        from contextlib import nullcontext
        from transformers.modeling_utils import ContextManagers
        try:
            from transformers.modeling_utils import no_init_weights
        except ImportError:
            from transformers.initialization import no_init_weights
        from hyper_parallel import init_empty_weights
        init_ctx = ContextManagers([no_init_weights(), init_empty_weights()]) if is_meta_device else nullcontext()
        with init_ctx:
            if checkpoint_path is not None:
                model = provider.build_model(config)
            else:
                build_from_config = getattr(provider, "build_from_config", provider.build_model)
                model = build_from_config(config)
        model = apply_model_infrastructure(
            model,
            mesh=distributed_setup.mesh_context,
            sharding_planner=planner,
            fsdp2_manager=fsdp,
            peft_config=peft_config,
            compile_config=compile_config,
            is_meta_device=is_meta_device,
            is_hf_model=True,
            device=_current_device(),
            load_base_model=checkpoint_path is not None,
            pretrained_path=checkpoint_path,
            validate_placement=validate_placement,
            distributed_setup=distributed_setup,
            activation_checkpoint=activation_checkpoint,
            swap_inputs=swap_inputs,
            activation_swap=activation_swap,
            model_init_dtype=model_init_dtype,
        )
        model.train()
        return model

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        model_provider: Any = None,
        model_type: Optional[str] = None,
        distributed_setup: Optional[DistributedSetup] = None,
        peft_config: Optional[Any] = None,
        compile_config: Optional[Union[CompileConfig, dict]] = None,
        activation_checkpoint: Optional[str] = None,
        swap_inputs: bool = False,
        activation_swap: str = "none",
        model_init_dtype: Optional[Literal["float16", "bfloat16", "float32"]] = None,
        validate_placement: bool = False,
        **kwargs: Any,
    ) -> torch.nn.Module:
        """Build a DiT model from a provider-owned config."""
        provider = _resolve_dit_provider(model_provider, model_type)
        if distributed_setup is None:
            raise ValueError("DiT AutoModel requires distributed_setup from BaseTrainer")
        prepare = getattr(provider, "prepare", None)
        if callable(prepare):
            prepare(distributed_setup, peft_config=peft_config, **kwargs)
        planner, fsdp = instantiate_infrastructure(
            distributed_setup=distributed_setup, device=_current_device()
        )
        is_meta_device = (
            torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        ) > 1
        from contextlib import nullcontext
        from transformers.modeling_utils import ContextManagers
        try:
            from transformers.modeling_utils import no_init_weights
        except ImportError:
            from transformers.initialization import no_init_weights
        from hyper_parallel import init_empty_weights
        init_ctx = ContextManagers([no_init_weights(), init_empty_weights()]) if is_meta_device else nullcontext()
        with init_ctx:
            build_from_config = getattr(provider, "build_from_config", provider.build_model)
            model = build_from_config(config, **kwargs)
        model = apply_model_infrastructure(
            model,
            mesh=distributed_setup.mesh_context,
            sharding_planner=planner,
            fsdp2_manager=fsdp,
            peft_config=peft_config,
            compile_config=compile_config,
            is_meta_device=is_meta_device,
            is_hf_model=True,
            device=_current_device(),
            load_base_model=False,
            validate_placement=validate_placement,
            distributed_setup=distributed_setup,
            activation_checkpoint=activation_checkpoint,
            swap_inputs=swap_inputs,
            activation_swap=activation_swap,
            model_init_dtype=model_init_dtype,
        )
        model.train()
        return model
