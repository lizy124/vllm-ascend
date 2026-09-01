#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Layerwise GVA transfer protocol (memcache backend).

GVA is a memcache-exclusive protocol, so this module owns the gate
derivation and the key formats, keeping the memcache-specific knowledge
out of the generic layers. Key formats are centralized here so the
worker-side and scheduler-side constructions cannot drift apart; the
strings are byte-for-byte identical to the pre-refactor ``pool_worker``
/ ``pool_scheduler`` implementations.
``tests/ut/distributed/ascend_store/test_gva_protocol.py`` locks the
gate truth table, the memcache exclusivity of the GVA store methods,
and the key formats with snapshot assertions.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def use_gva_layerwise(use_layerwise: bool, extra_config: Mapping[str, Any]) -> bool:
    """Single derivation point for the GVA layerwise transfer mode.

    ``use_layerwise`` is the caller's authoritative flag (a constructor
    parameter on the worker/scheduler; the layout path reads it from
    the config itself). The backend name — its config key, default,
    and normalization (strip + lower) — is owned here, so the
    memcache-specific knowledge stays single-sourced. The layerwise GVA
    fast path is a memcache-specific protocol, so every call site must
    derive the flag from here instead of re-spelling the comparison.
    Duplicated derivations have already caused a live regression:
    #14465 deleted one copy as dead code while a reader still consumed
    it.
    """
    backend_name = str(extra_config.get("backend", "mooncake")).strip().lower()
    return use_layerwise and backend_name == "memcache"


class GVAKeyFactory:
    """String formats for the layerwise GVA keys.

    Single-group models use the PR #11585 format (model@hash@rank) for
    backward compatibility. Multi-group models include group_id
    (model@group_id@hash@rank) to distinguish groups.
    """

    @staticmethod
    def full_key(
        model_name: str,
        group_id: int,
        block_hash_hex: str,
        head_or_tp_rank: int,
        num_groups: int,
    ) -> str:
        if num_groups > 1:
            return f"{model_name}@{group_id}@{block_hash_hex}@{head_or_tp_rank}"
        else:
            return f"{model_name}@{block_hash_hex}@{head_or_tp_rank}"

    @staticmethod
    def partial_key(
        model_name: str,
        req_id: str,
        group_id: int,
        block_index: int,
        end_token: int,
        head_or_tp_rank: int,
    ) -> str:
        return f"{model_name}@partial@{req_id}@{group_id}@{block_index}@{end_token}@{head_or_tp_rank}"

    @staticmethod
    def hit_check_keys(
        model_name: str,
        group_id: int,
        block_hash_hex: str,
        num_ranks: int,
        num_groups: int,
    ) -> list[str]:
        """All-rank GVA keys for scheduler-side hit check.

        Returns one key per head_or_tp_rank (ranks in the same put_step
        group share one key for MLA).
        """
        if num_groups > 1:
            return [f"{model_name}@{group_id}@{block_hash_hex}@{h}" for h in range(num_ranks)]
        else:
            return [f"{model_name}@{block_hash_hex}@{h}" for h in range(num_ranks)]
