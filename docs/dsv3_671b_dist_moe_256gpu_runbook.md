# DeepSeek V3 671B DistMoE 256-GPU Runs

This handoff runs two 256-GPU experiments directly from one self-contained
Conda fbpkg. Do not clone TorchTitan or install PyTorch, TorchAO,
FlashAttention-4, DistMoE, or any Python dependency separately. The fbpkg
contains the complete runtime, TorchTitan source, recipes, tokenizer, and
analysis tools required by the runs.

## Runtime artifact and source reports

Use this exact immutable package:

```text
torchtitan_conda_ivankobzarev_dist_moe_256gpu_20261001_v1:1f35de49f9034a8688be3026a2a384e5
```

It has a 21.33 GiB logical size and expires on 2026-10-29 at 07:04 Pacific
time. Stop and request a newly published and reverified UUID if that expiry
does not cover the runs and artifact collection. Do not substitute `LATEST`.

For provenance only, the package contains TorchTitan revision
`70468662c2047beff4f3920ead5d84c00658cfad`. A Git checkout is not needed to
fetch, verify, or run the package.

- Chien-Chin source: [P2483940224](https://www.internalfb.com/phabricator/paste/view/P2483940224?view=markdown)
- Sanket source: [P2527793677](https://www.internalfb.com/phabricator/paste/view/P2527793677?view=markdown)

The Chien-Chin recipe is the current-stack expression of ladder1 R4. The
Sanket-derived recipe keeps PP2/VPP8, DP128, EP64, expert-FSDP2, MBS1/LBS32,
RAF-never, and the asymmetric 16-stage split. It intentionally omits MTP1 and
eager in-place WGrad accumulation: current public TorchTitan rejects MTP with
pipeline parallelism and reserves in-place WGrad accumulators for
GraphRuntime. Therefore, compare the second run's topology and stability with
P2527793677, but do not treat its throughput as an exact reproduction.

## Fetch and verify the fbpkg

Fetch the same immutable UUID into a shared path visible at the same absolute
location on every worker:

```bash
export FBPKG_ID='torchtitan_conda_ivankobzarev_dist_moe_256gpu_20261001_v1:1f35de49f9034a8688be3026a2a384e5'
export FBPKG_DEST=/absolute/shared/path/torchtitan_dist_moe_runtime
mkdir -p "$FBPKG_DEST"
fbpkg fetch "$FBPKG_ID" --dest "$FBPKG_DEST" --extract --verify
export RUNTIME_ROOT="$FBPKG_DEST/conda"
source "$RUNTIME_ROOT/bin/activate"
cd "$RUNTIME_ROOT/src/torchtitan"
cat "$RUNTIME_ROOT/torchtitan_fbpkg_provenance/revisions.txt"
```

Run the dependency and recipe preflight on a GB300 worker:

```bash
python - <<'PY'
from pathlib import Path
import importlib.metadata as metadata
import sys

import dist_moe
import dist_moe._blockscaled  # noqa: F401
import functorch
import torch
import torchao
import torchtitan
from dist_moe import BlockScaledFormat, DistMoeBlockScaledConfig
from torch.utils.checkpoint import _is_cacheable_effect

prefix = Path(sys.prefix).resolve()
for module in (functorch, torch, torchao, dist_moe, torchtitan):
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(prefix), (module.__name__, path, prefix)

assert torch.cuda.get_device_capability() == (10, 3)
assert torch.cuda.get_device_properties(0).total_memory >= 250 * 1024**3
assert hasattr(torch.ops.aten, "_scaled_addmm_")
assert hasattr(torch.ops.dist_moe, "block_scaled_backward_accumulate")
assert hasattr(torch.ops.dist_moe, "bf16_backward_accumulate")
assert torch._C._dispatch_has_kernel_for_dispatch_key(
    "torchao::mxfp8_quantize", "CUDA"
)
assert BlockScaledFormat.MXFP8_E4M3
assert DistMoeBlockScaledConfig
assert _is_cacheable_effect
print("torch", torch.__version__, torch.version.git_version)
print("torchao", metadata.version("torchao"), torchao.__file__)
print("dist_moe", dist_moe.__file__)
print("torchtitan", torchtitan.__file__)
PY

pytest -q tests/unit_tests/cpu/test_dist_moe.py \
  -k 'chien_chin_256gpu or sanket_topology_256gpu'
tlparse --version
```

Stop if any check fails.

## Common launch contract

Launch exactly 256 workers, one process per GB300 GPU. The launcher must set
valid `RANK`, `WORLD_SIZE=256`, `LOCAL_RANK`, `MASTER_ADDR`, and `MASTER_PORT`
for every worker. The reference packing is 64 hosts with four GPUs per host;
report any different packing and keep each EP64 group within a supported
high-bandwidth fabric domain.

Use one shared output root and a unique node-local cache root on every host:

```bash
export OUTPUT_ROOT=/absolute/shared/path/to/run_outputs
export LOCAL_CACHE_ROOT=/absolute/node_local/path/to/cache
export TRITON_CACHE_DIR="$LOCAL_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_CACHE_ROOT/torchinductor"
export CUTE_DSL_CACHE_DIR="$LOCAL_CACHE_ROOT/cute_dsl"
export CUDA_CACHE_PATH="$LOCAL_CACHE_ROOT/cuda"
mkdir -p "$OUTPUT_ROOT" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$CUTE_DSL_CACHE_DIR" "$CUDA_CACHE_PATH"
```

For each experiment, steps 1-10 are warmup, steps 11-40 are the 30 measured
steps, and step 41 is the profiler and memory-snapshot step. Do not include
step 41 in performance statistics. Only rank zero writes `TORCH_TRACE`.

## Run 1: Chien-Chin ladder1 R4

This run uses PP1, DP/FSDP256, EP64, expert-FSDP4, TP1, CP1, local batch 1,
gradient accumulation 16, RAF-never, dense symmetric-memory FSDP, deferred
gradient reduction, first-microbatch unshard, last-microbatch reduce-grad, and
GraphTrainer WGrad producer fusion.

Launch all 256 workers with:

```bash
export RUN_OUTPUT="$OUTPUT_ROOT/chien_chin_r4"
if [[ "$RANK" -eq 0 ]]; then
  export TORCH_TRACE="$RUN_OUTPUT/tlparse_raw"
else
  unset TORCH_TRACE
fi
python -u -m torchtitan.train \
  --output-dir "$RUN_OUTPUT" \
  --module torchtitan_recipes.graph_trainer.deepseek_v3 \
  --config graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_256gpu_profile
```

The historical target from P2483940224 is `5,880.6 +/- 13.8` tokens/s/GPU,
about 1.51 million aggregate tokens/s. The new runtime may differ.

## Run 2: Sanket-derived PP2/VPP8 topology

This run uses PP2 with eight virtual stages per rank, DP/FSDP128, EP64,
expert-FSDP2, TP1, CP1, 32 pipeline microbatches, MBS1/LBS32/GBS4096,
RAF-never, no symmetric-memory FSDP, automatic unshard lookahead, and at most
eight active unsharded stages per pipeline rank.

The recipe uses the production C4 dataset entry. Ensure the workers can access
that dataset before allocating 256 GPUs. Launch all workers with:

```bash
export RUN_OUTPUT="$OUTPUT_ROOT/sanket_pp2_vpp8"
if [[ "$RANK" -eq 0 ]]; then
  export TORCH_TRACE="$RUN_OUTPUT/tlparse_raw"
else
  unset TORCH_TRACE
fi
python -u -m torchtitan.train \
  --output-dir "$RUN_OUTPUT" \
  --module torchtitan_recipes.models.deepseek_v3 \
  --config deepseek_v3_671b_dist_moe_mxfp8_sanket_topology_256gpu_profile
```

P2527793677 reported about 5,611 tokens/s/GPU for its MTP1 fused candidate.
That number is context only because this current-stack adaptation omits MTP1
and eager in-place WGrad accumulation.

## Process and report artifacts

Render rank-zero `TORCH_TRACE` data for each experiment:

```bash
tlparse "$RUN_OUTPUT/tlparse_raw" -o "$RUN_OUTPUT/tlparse_html" \
  --no-browser --overwrite
```

The profiler writes traces under:

```text
$RUN_OUTPUT/profiling/traces/iteration_41/rank<RANK>_trace.json.gz
```

Share rank 0 from both experiments and rank 128 from the PP2 experiment. From
a machine with fbsource, internal `fbpython`, and Manifold access:

```bash
export FBSOURCE_ROOT=/absolute/path/to/fbsource
export SHARE_TRACE="$FBSOURCE_ROOT/arvr/scripts/perfetto/share_trace.py"
test -x "$SHARE_TRACE"

for TRACE_PATH in \
  "$OUTPUT_ROOT/chien_chin_r4/profiling/traces/iteration_41/rank0_trace.json.gz" \
  "$OUTPUT_ROOT/sanket_pp2_vpp8/profiling/traces/iteration_41/rank0_trace.json.gz" \
  "$OUTPUT_ROOT/sanket_pp2_vpp8/profiling/traces/iteration_41/rank128_trace.json.gz"; do
  test -f "$TRACE_PATH"
  "$SHARE_TRACE" "$TRACE_PATH" | tee "$TRACE_PATH.share.txt"
done
```

`share_trace.py` uses a 28-day TTL by default and is intentionally not bundled
in the portable fbpkg because it requires an internal fbsource environment.

For each run, report:

1. The exact fbpkg UUID, provenance files, commands, GPU count/model, and
   host/GPU packing.
2. Per-GPU tokens/s and MFU for every measured step 11-40, plus mean and
   standard deviation. Report aggregate tokens/s as the per-GPU mean times
   256. If MFU is unavailable, report TFLOP/s.
3. Peak allocated and reserved GPU memory, preferably min/median/max across
   all ranks, plus the step-41 memory snapshots.
4. All logged loss and gradient-norm values; they must remain finite and the
   gradient norm must remain nonzero.
5. Complete stdout/stderr from every rank, including unabridged error logs on
   failure, raw and rendered `tlparse`, all raw profiler traces, and the three
   `share_trace.py` URLs above.

Both runs must finish all 41 steps without OOM, NaN, graph recapture, or
distributed errors.
