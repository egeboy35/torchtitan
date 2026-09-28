# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel transform."""

from collections.abc import Mapping
from dataclasses import dataclass

from torchtitan.models.common.attention import BaseAttention, InnerAttention
from torchtitan.models.common.cp_attention import CPInnerAttention
from torchtitan.protocols.module import Module

from .base import convert_config_type, ModelConfigTransform, ModelConfigTransformContext

__all__ = ["ContextParallelTransform"]


@dataclass(kw_only=True, slots=True)
class ContextParallelTransform(ModelConfigTransform):
    """Replace each inner-attention type with its configured CP backend."""

    inner_attention_backends: Mapping[type[InnerAttention], type[InnerAttention]]
    """Map each source inner-attention type to its CP replacement."""

    def __post_init__(self) -> None:
        if not self.inner_attention_backends:
            raise ValueError("inner_attention_backends must not be empty.")
        for source, replacement in self.inner_attention_backends.items():
            if not issubclass(source, InnerAttention):
                raise ValueError(f"{source.__qualname__} must inherit InnerAttention.")
            if not issubclass(replacement, CPInnerAttention):
                raise ValueError(
                    f"{replacement.__qualname__} must inherit CPInnerAttention."
                )

    def transform(
        self,
        model: Module.Config,
        *,
        context: ModelConfigTransformContext | None = None,
    ) -> Module.Config:
        del context
        for _, traversed, _, _ in model.traverse(BaseAttention.Config):
            attention = traversed
            source = attention.inner_attention._owner
            assert source is not None and issubclass(source, InnerAttention)
            if source not in self.inner_attention_backends:
                raise ValueError(
                    "No context-parallel backend configured for "
                    f"{source.__qualname__}."
                )
            attention.inner_attention = convert_config_type(
                attention.inner_attention,
                self.inner_attention_backends[source],
            )
        return model
