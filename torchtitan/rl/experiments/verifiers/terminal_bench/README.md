# Terminal-Bench with Verifiers and TitanRL

This experiment trains a Qwen3.5 terminal agent (9B or 35B-A3B) on a Harbor
dataset, and validates it on all 89 Terminal-Bench 2.1 tasks.
TitanRL schedules rollout groups, generates tokens, and trains the model;
Verifiers 0.3.1 runs the
Terminus-2 agent and the Harbor task verifier. It is an experiment, not a new
TitanRL rollout or sandbox backend.

This experiment uses the existing TitanRL and Verifiers bridge in
`torchtitan/rl/examples/verifiers/` on `main`. Only the task-specific adapter,
recipe, and tests live here; TitanRL's core is unchanged.

## Datasets

Install the TorchTitan RL dependencies and this example's `requirements.txt` in
the same Python 3.12 environment. Verifiers 0.3.1 requires MCP 1.x; use a
separate environment if another project has installed MCP 2.x. Install Docker
on the controller host and make sure the user can run Docker containers. The
agent runs **inside** the task's container; `SubprocessConfig` must not be used
for untrusted terminal tasks. Verifiers pulls the task's declared image and
uploads grading files only after the agent finishes.

Tasks come from Harbor datasets selected by id, the usual Verifiers way: the
`harbor` CLI downloads `org/name` or `org/name@ref` into `~/.cache/harbor` the
first time it is needed. Pin `@ref` (a tag, revision or digest) and record it
alongside the model checkpoint; the 2.0 and 2.1 benchmarks are not
interchangeable. Two variables select the datasets:

- `TERMINAL_BENCH_EVAL_DATASET`: the Terminal-Bench 2.1 dataset, 89 tasks, used
  for validation at the start and end of training.
- `TERMINAL_BENCH_TRAIN_DATASET`: a separate Harbor dataset to train on. It must
  not be the benchmark. This example does not fetch, filter, evolve, or publish
  training examples.

Each task must declare a pullable `[environment].docker_image`; a task with only
a Dockerfile is rejected rather than silently evaluated in the wrong
environment. The published images do not all ship `tmux`. Harbor's Terminus-2
session installs it when it is missing, which needs network in the task
container, just as scoring does:

**The task container needs outbound network at scoring time, and removing it
zeroes the whole benchmark silently.** Every Terminal-Bench 2.1 task ends its
`tests/test.sh` by installing its own toolchain -- `apt-get`, then `uv` from
astral.sh, then pytest and the task's pinned libraries from PyPI -- against a
published image that ships none of it. That is a property of the 89 tasks, not
of the runtime. Docker's default bridge network provides egress, so the
default configuration here works; a network-isolated runtime, a hardened
Docker config, or a host without egress breaks it. The failure is invisible:
each `test.sh` ends in `if ...; then echo 1; else echo 0; fi`, which exits 0
whether the tests passed, failed, or were never installed, so the grader
writes a `0` and Harbor returns it as a real score. Every rollout of every run
then reports `reward=0.000` while looking perfectly healthy, for every model
and every configuration. The 12,000 s `scoring` timeout below is sized for
those installs.

```bash
export TERMINAL_BENCH_TRAIN_DATASET=org/train-tasks@<ref>
export TERMINAL_BENCH_EVAL_DATASET=org/terminal-bench-2-1@<ref>

python -m torchtitan.rl.train \
  --module torchtitan.rl.experiments.verifiers.terminal_bench \
  --config rl_grpo_qwen35_9b_terminal_bench \
  --hf_assets_path /path/to/Qwen3.5-9B
```

The two dataset ids must differ, and the selected task IDs should be disjoint.
The recipe uses 8 trainer GPUs (FSDP) and 8 one-GPU generator replicas by
default (16 GPUs total); adjust GPU counts for the available hosts. TitanRL's
CLI defaults to a single host: placing these meshes across hosts requires a
caller-provided `HostMeshes` launcher. Mainline TitanRL cannot configure eight
independent vLLM DP replicas inside one generator without expert parallelism,
so the generator layout is not identical to the colleague branch's single 8-DP
generator. It uses a 65,536-token model/trainer context, up to 16,384 generated
tokens per turn, 120 agent turns, 32 siblings per group, 8 groups per training
step, 4 target off-policy steps, 100 training steps, and a 1e-6 constant
learning rate. It keeps fp32 master weights/Adam states with bf16 FSDP compute,
full activation checkpointing, and 20-step DCP saves. Historical thinking is
retained across turns. Verifiers uses the Terminus-2 scaffold with Harbor
0.22.0 and grades in the agent's own container. The binary task reward flows
through TitanRL's ordinary advantage and GRPO training path.

`TrainingSampleBuilder.drop_zero_std_reward_groups` defaults to `True`, and the
reward here is binary. A group whose 32 samples all score 0 has no advantage
spread and is discarded, which is the right behaviour for training -- but with
a base model that solves nothing, *every* group is discarded, no batch is ever
formed, and the run sits in `wait_for_training_batch` indefinitely without
reaching a second step. It looks like slow rollouts rather than a filter. The
in-tree integration test at `tests/integration_tests/rl.py` sets the flag to
`False` for the same reason. Check the base model's pass rate on a handful of
tasks before concluding that the pipeline is slow. Its XML parser, disabled context
summarization, and 120-turn limit match the colleague's Terminus-2 setup.
The example configures these three options on Verifiers' upstream program;
Terminus-2 itself is not forked. Timeouts are 7,200 seconds for the
agent and 12,000 seconds for scoring; the task's authored timeouts are ignored
to avoid prematurely cutting off slow inference.

## Model sizes

Both sizes share the datasets, sampling, loop and optimizer settings above.
Only the checkpoint, the trainer precision, the trainer and generator layouts
and the generator CUDA graph mode change with the model. Each recipe uses 16
GPUs: 8 for the trainer and 8 for generators.

| Model | Config | Trainer | Generators | Trainer precision | Generator CUDA graphs |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-9B | `rl_grpo_qwen35_9b_terminal_bench` | FSDP=8 | 8 x 1 GPU | fp32 master weights | `FULL_DECODE_ONLY` |
| Qwen3.5-35B-A3B | `rl_grpo_qwen35_35b_a3b_terminal_bench` | FSDP=4, TP=2, EP=8 | 2 x 4 GPUs (DP=2, TP=2, EP=4) | full bf16 | off |

- The 35B-A3B layout follows the model shape. It has 2 KV heads and 256 experts:
  TP is capped at 2, trainer EP must be at least TP and divide both the experts
  and `dp_shard * tp`, and the generator's DP axis exists only to supply
  expert-parallel ranks, so its EP equals DP x TP.
- Full bf16 (`dtype="bfloat16"`) keeps parameters, gradients and optimizer
  states in bf16 with no fp32 copy. It puts the 35B-A3B model states at roughly
  35 GB per trainer GPU; fp32 master weights would need about 70 GB before
  activations. The 9B keeps fp32 master weights. At a 1e-6 learning rate, bf16
  parameters can round small updates away, so prefer fp32 master weights where
  the memory allows.
- The 35B-A3B generator runs without CUDA graphs. The standard MoE token
  dispatcher copies the all-to-all split sizes to the host, and graph capture
  fails on that copy. Re-enable capture together with a dispatcher that avoids
  the host read, such as HybridEP with `non_blocking_capacity_factor`.
- The 35B-A3B recipe has not been run end to end. The CPU tests check that the
  layout is consistent with the model's KV heads and experts; they do not check
  that it fits in memory or that rollouts complete.

## Fidelity and open dependencies

The 9B recipe preserves the colleague branch's Qwen3.5-9B model choice; the
35B-A3B recipe is an addition to it. The example preserves the
Terminus-2 v0.22 scaffold, 120-turn budget, Harbor task instructions, isolated
per-task environment, in-place verifier, and 0/1 reward. Verifiers reads
binary grading fixtures directly from the Harbor task package; it does not need
the branch's base64-encoded JSONL conversion.

These parts do **not** currently reproduce that branch's measured run:

- Verifiers 0.3.1 supports Docker and Prime runtimes but not Daytona. A
  Daytona backend is a TODO for the Verifiers repository; integrate and test it
  separately before comparing sandbox resource limits or rollout throughput.
- Dockerfile-only tasks need a prebuilt, pullable image; Verifiers does not
  build task Dockerfiles. The colleague's benchmark used derived images with
  `tmux` verified in every task environment; here Harbor installs `tmux` at run
  time when the published image lacks it.
- Datasets are fetched by Harbor id into `~/.cache/harbor`. A cluster without
  access to the Harbor Hub can pre-populate that cache with an exported task
  tree; loading from an arbitrary local path is not supported here.
- Verifiers' built-in Terminus-2 config does not expose the XML parser and
  summary policy. `harness.py` configures the upstream program for this
  experiment; upstreaming those knobs to Verifiers would remove the adapter.
- Harbor grading returns `0.0` when the grader never ran, indistinguishable
  from a task the agent failed. `HarborTask._graded` discards `test.sh`'s
  result, and both reward-file readers swallow their exceptions and fall back
  to zero, so "tests ran and the agent failed", "tests never ran", "reward file
  missing" and "reward file unparsable" are one value. Distinguishing a
  missing reward file from one containing `0` would separate the infrastructure
  cases without needing anything from task scripts; the exit code alone would
  not, because every TB2.1 `test.sh` ends in an `if`/`echo` that exits 0
  regardless. Until that lands, a run of uniform zeros here should be treated
  as unexplained rather than as a capability measurement.
- TitanRL does not read `Trace.metrics`. Verifiers implements that channel
  fully -- a `@metric` decorator whose functions are collected around every
  rollout -- but the adapter in `torchtitan/rl/examples/verifiers` consumes
  only `generation_metadata.metrics`, and `Rollout` has no field to hold the
  rest. Per-rollout metrics emitted by a taskset therefore cross the process
  boundary and are dropped without warning. Anything this experiment wants to
  report per rollout needs its own delivery path today.
- Verifiers' Harbor fixture staging restores host file ownership, which fails
  in a rootless container with unmapped host IDs. This experiment does not work
  around it; it should be fixed in Verifiers' shared Harbor task implementation.
- Mainline TitanRL supports GRPO, while the colleague's production recipe uses
  DPPO with a different trust-region loss and dedicated async scheduling.
  Mainline also lacks its per-group KV cache salting on weight sync, independent
  vLLM DP for this dense model, dedicated async validation generators, and
  mid-training periodic validation. Do not add experiment-only branches to
  TitanRL core to emulate these features.
- TitanRL's current validation reports greedy pass@1. The colleague's
  Terminal-Bench runs also report pass@5; these metrics cannot be compared
  directly without a validated multi-sample evaluation path.
- Container runtime details, agent transcripts, and model/token-level
  numerics still need a live Docker + GPU comparison against a pinned
  colleague-branch checkpoint and the same frozen 2.1 task revision. This
  example does not claim identical reward curves or loss without that run.

Only the recipe and CPU checks belong to this upstream change.
The colleague branch's evolution loop, curated task outputs, runbooks, logs,
and training-data artifacts are intentionally out of scope.
