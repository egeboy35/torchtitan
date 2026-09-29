# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 terminal-agent recipes using Verifiers and TitanRL."""

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from renderers import Qwen35RendererConfig

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.renderer import from_renderers
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.config.transform import LMHeadCastConverter
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3_5 import model_registry
from torchtitan.rl.controller import AsyncLoopConfig, ValidationConfig
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.distributed.routing.inter_generator import InterGeneratorRouter
from torchtitan.rl.distributed.routing.strategies import (
    LeastLoadedRoutingStrategy,
    StickySessionRoutingStrategy,
)
from torchtitan.rl.experiments.verifiers.terminal_bench.controller import (
    TerminalBenchController,
)
from torchtitan.rl.experiments.verifiers.terminal_bench.rollouter import (
    terminal_bench_rollouter_config,
)
from torchtitan.rl.generator import SamplingConfig, VLLMCudaGraphConfig, VLLMGenerator
from torchtitan.rl.losses import GRPOLoss
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.trainer import Trainer


@dataclass(frozen=True, kw_only=True)
class _ModelRecipe:
    """The model-dependent part of a Terminal-Bench recipe.

    Task trees, sampling, the optimizer and the async loop are shared by every
    model. Only the checkpoint, the trainer precision, the trainer and generator
    layouts and the generator CUDA graph mode change with the model.
    """

    flavor: str
    """Qwen3.5 model flavor passed to ``model_registry``."""

    checkpoint_name: str
    """Directory name of the HF checkpoint under ``torchtitan/rl/example_checkpoint``."""

    dump_name: str
    """Output directory stem under ``outputs/rl``."""

    trainer_dtype: Literal["bfloat16", "float32"]
    """``float32`` keeps fp32 master weights; ``bfloat16`` trains fully in bf16."""

    trainer_parallelism: ParallelismConfig
    generator_parallelism: InferenceParallelismConfig
    num_generators: int
    cuda_graph_mode: Literal["NONE", "FULL_DECODE_ONLY", "FULL"]


def _qwen35_9b() -> _ModelRecipe:
    """16 GPUs: 8 trainer (FSDP=8) and 8 single-GPU generators."""
    return _ModelRecipe(
        flavor="9B",
        checkpoint_name="Qwen3.5-9B",
        dump_name="qwen35_9b",
        trainer_dtype="float32",
        trainer_parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=8,
            tensor_parallel_degree=1,
        ),
        generator_parallelism=InferenceParallelismConfig(
            data_parallel_degree=1,
            tensor_parallel_degree=1,
        ),
        num_generators=8,
        cuda_graph_mode="FULL_DECODE_ONLY",
    )


def _qwen35_27b() -> _ModelRecipe:
    """16 GPUs: 8 trainer (FSDP=4 x TP=2) and 2 generators of 4 GPUs (TP=4).

    Both TP degrees divide the 4 KV heads. The trainer trains fully in bf16:
    model states take about 8 bytes per parameter, roughly 27 GB per GPU across
    8 GPUs, where fp32 master weights would need about 54 GB per GPU before any
    activations.
    """
    return _ModelRecipe(
        flavor="27B",
        checkpoint_name="Qwen3.5-27B",
        dump_name="qwen35_27b",
        trainer_dtype="bfloat16",
        trainer_parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=4,
            tensor_parallel_degree=2,
        ),
        generator_parallelism=InferenceParallelismConfig(
            data_parallel_degree=1,
            tensor_parallel_degree=4,
        ),
        num_generators=2,
        cuda_graph_mode="FULL_DECODE_ONLY",
    )


def _qwen35_35b_a3b() -> _ModelRecipe:
    """16 GPUs: 8 trainer (FSDP=4 x TP=2, EP=8) and 2 generators of 4 GPUs.

    The MoE layout is constrained from both sides:

    - The model has 2 KV heads, so TP cannot exceed 2 in either role.
    - Trainer EP must be at least TP, divide the 256 experts and divide
      ``dp_shard * tp``. EP=8 spans the whole sparse region, so each rank holds
      32 experts.
    - The generator's DP axis exists only to supply ranks for expert
      parallelism, and its EP must equal DP x TP: DP=2, TP=2, EP=4, 64 experts
      per rank.

    The trainer trains fully in bf16, about 35 GB of model states per GPU across
    8 GPUs, where fp32 master weights would need about 70 GB.

    Generator CUDA graphs are off. The standard MoE token dispatcher copies the
    all-to-all split sizes to the host, which CUDA graph capture does not allow
    ("Cannot copy between CPU and CUDA tensors during CUDA graph capture"). Turn
    capture back on together with a dispatcher that avoids the host read, such as
    HybridEP with ``non_blocking_capacity_factor``.
    """
    return _ModelRecipe(
        flavor="35B-A3B",
        checkpoint_name="Qwen3.5-35B-A3B",
        dump_name="qwen35_35b_a3b",
        trainer_dtype="bfloat16",
        trainer_parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=4,
            tensor_parallel_degree=2,
            expert_parallel_degree=8,
        ),
        generator_parallelism=InferenceParallelismConfig(
            data_parallel_degree=2,
            tensor_parallel_degree=2,
            expert_parallel_degree=4,
        ),
        num_generators=2,
        cuda_graph_mode="NONE",
    )


def _tasks_root(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Set {name} to a frozen Harbor task-tree directory")
    return Path(value)


def _image_overrides_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def _terminal_agent_config(
    *,
    recipe: _ModelRecipe,
    train_tasks_root: Path,
    eval_tasks_root: Path,
    eval_only: bool,
) -> TerminalBenchController.Config:
    max_context_length = 65536
    model_config = model_registry(
        recipe.flavor,
        seq_len=max_context_length,
        attn_backend="varlen",
        converters=[LMHeadCastConverter.Config()],
    )
    return TerminalBenchController.Config(
        eval_only=eval_only,
        model=model_config,
        hf_assets_path=f"torchtitan/rl/example_checkpoint/{recipe.checkpoint_name}",
        dump_folder=f"outputs/rl/{recipe.dump_name}_terminal_bench",
        async_loop=AsyncLoopConfig(
            num_training_steps=0 if eval_only else 100,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=32,
            target_offpolicy_steps=4,
            validation=ValidationConfig(num_samples=89),
        ),
        rollouter=terminal_bench_rollouter_config(
            train_tasks_root,
            eval_tasks_root,
            train_images_path=(
                _image_overrides_path("TERMINAL_BENCH_EVAL_IMAGES")
                if eval_only
                else _image_overrides_path("TERMINAL_BENCH_TRAIN_IMAGES")
            ),
            validation_images_path=_image_overrides_path("TERMINAL_BENCH_EVAL_IMAGES"),
            eval_only=eval_only,
        ),
        renderer=from_renderers(
            Qwen35RendererConfig(
                enable_thinking=True,
                thinking_retention="all",
            )
        ),
        num_generators=recipe.num_generators,
        generator_router=InterGeneratorRouter.Config(
            strategy=StickySessionRoutingStrategy.Config(
                fallback_strategy=LeastLoadedRoutingStrategy.Config()
            )
        ),
        metrics=MetricsProcessor.Config(
            console_log_keys_validation=[
                "validation_reward/_mean",
                "validation_reward/_max",
                "timing/validate",
            ],
        ),
        trainer=Trainer.Config(
            optimizer=default_adamw(
                lr=1e-6,
                betas=(0.9, 0.999),
                weight_decay=0.0,
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=0,
                min_lr_factor=1.0,
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=max_context_length,
                max_context_length=max_context_length,
                dtype=recipe.trainer_dtype,
            ),
            parallelism=recipe.trainer_parallelism,
            activation_checkpoint=FullAC.Config(),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=20,
                keep_latest_k=3,
                async_mode="async",
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=32,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(model_config)
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            cuda_graph=VLLMCudaGraphConfig(mode=recipe.cuda_graph_mode),
            parallelism=recipe.generator_parallelism,
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=16384,
            ),
        ),
    )


def _train_config(recipe: _ModelRecipe) -> TerminalBenchController.Config:
    return _terminal_agent_config(
        recipe=recipe,
        train_tasks_root=_tasks_root("TERMINAL_BENCH_TRAIN_TASKS_ROOT"),
        eval_tasks_root=_tasks_root("TERMINAL_BENCH_EVAL_TASKS_ROOT"),
        eval_only=False,
    )


def _eval_config(recipe: _ModelRecipe) -> TerminalBenchController.Config:
    eval_root = _tasks_root("TERMINAL_BENCH_EVAL_TASKS_ROOT")
    config = _terminal_agent_config(
        recipe=recipe,
        train_tasks_root=eval_root,
        eval_tasks_root=eval_root,
        eval_only=True,
    )
    checkpoint = os.environ.get("TERMINAL_BENCH_CHECKPOINT")
    if checkpoint:
        config.trainer = replace(
            config.trainer,
            checkpointer=replace(
                config.trainer.checkpointer,
                initial_load_path=checkpoint,
                initial_load_in_hf=False,
                initial_load_model_only=True,
            ),
        )
    return config


def rl_grpo_qwen35_9b_terminal_bench() -> TerminalBenchController.Config:
    """Train on frozen terminal tasks and validate on Terminal-Bench 2.1."""
    return _train_config(_qwen35_9b())


def rl_grpo_qwen35_9b_terminal_bench_eval() -> TerminalBenchController.Config:
    """Score the 89 Terminal-Bench 2.1 tasks without optimizer steps."""
    return _eval_config(_qwen35_9b())


def rl_grpo_qwen35_27b_terminal_bench() -> TerminalBenchController.Config:
    """Qwen3.5-27B dense: train on frozen tasks, validate on Terminal-Bench 2.1."""
    return _train_config(_qwen35_27b())


def rl_grpo_qwen35_27b_terminal_bench_eval() -> TerminalBenchController.Config:
    """Score the 89 Terminal-Bench 2.1 tasks with Qwen3.5-27B, no optimizer steps."""
    return _eval_config(_qwen35_27b())


def rl_grpo_qwen35_35b_a3b_terminal_bench() -> TerminalBenchController.Config:
    """Qwen3.5-35B-A3B MoE: train on frozen tasks, validate on Terminal-Bench 2.1."""
    return _train_config(_qwen35_35b_a3b())


def rl_grpo_qwen35_35b_a3b_terminal_bench_eval() -> TerminalBenchController.Config:
    """Score the 89 Terminal-Bench 2.1 tasks with Qwen3.5-35B-A3B, no optimizer steps."""
    return _eval_config(_qwen35_35b_a3b())
