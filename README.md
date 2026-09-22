# qwen27b-kernel-bench

Shape census and kernel level benchmarks for Qwen3.8-27B on a single NVIDIA H200, with vLLM 0.29 as the engine of record. I planned this for a B300, but Verda had none in stock, so the numbers come from an H200 141GB. The method is the same on any GPU, because every ceiling is measured on the device.

Qwen3.8-27B is a hybrid: 48 Gated DeltaNet (GDN) layers and 16 full attention layers. This repo answers three questions about it:

1. Which operators does the model run, at which shapes, and how are those shapes distributed under real continuous batching?
2. How do the two mixers behave at kernel level when the cache is warm (most KV already cached) and when the batch is ragged (many different sequence lengths in one batch)?
3. Do the measured kernel times add up to the step time the engine reports, and how large is the part they do not explain?

Every utilization number is relative to a ceiling measured on the same GPU: achieved HBM bandwidth and achieved FP8 and BF16 GEMM throughput. Spec sheet numbers are listed for reference only.

## Model geometry

| | Value |
|---|---|
| Layers | 64 = 16 x (3 GDN + 1 full attention); attention at layers 3, 7, ..., 63 |
| Hidden / MLP / vocab | 5120 / 17408 / 248320 |
| Attention | 24 query heads, 4 KV heads, head_dim 256, output gate, rotary on 64 dims |
| GDN | 16 key heads, 48 value heads, head_dim 128, conv kernel 4 over 10240 channels |
| GDN state per sequence per layer | recurrent (48, 128, 128) fp32 = 3 MiB, conv (10240, 3) bf16 |
| GEMMs as vLLM runs them | GDN `in_proj_qkvz` 5120 to 16384, `in_proj_ba` 5120 to 96, `out_proj` 6144 to 5120; attention `qkv_proj` 5120 to 14336, `o_proj` 6144 to 5120; MLP `gate_up_proj` 5120 to 34816, `down_proj` 17408 to 5120 |
| KV cache in vLLM | 784 token blocks; on the H200 FlashAttention 3 runs directly on those blocks |

## What gets measured

The shape census has three sources, and they are diffed against each other:

- analytic: every operator and shape derived from the model config, plus how the shapes change with tensor parallel 1/2/4/8, fp8 KV cache, speculative decoding and prefix caching
- observed: torch profiler traces from a live vLLM server, once in eager mode for the operator shapes and once in the default compiled CUDA graph mode for the kernels that run in production
- real distribution: prefill tokens, decode tokens and sequence count of every engine step under load, with prefix caching off and on

The kernel benchmarks cover the attention and GDN paths vLLM calls: paged prefill, append and decode attention, causal conv1d, chunked gated delta rule and fused recurrent decode. Both mixers run in the same regimes:

| Regime | Definition |
|---|---|
| Cold | no cached tokens; prefill over 128 to 65536 tokens, decode over KV 1k to 128k |
| Warm, fraction view | context L in multiples of 7840 tokens, cached fraction 0 / 0.5 / 0.9 / 0.99 in whole 784 token blocks, only the new tokens are computed |
| Warm, fixed new view | 64 / 512 / 2048 new tokens over cached history from 0 to 128k |
| Ragged prefill | equal total new tokens, lengths uniform / 80 to 100 percent jitter / lognormal / bimodal / mixed prefill plus decode |
| Ragged decode | one new token per sequence, KV lengths uniform / jitter / lognormal / bimodal at equal total KV |

For GDN, warm means a restored fixed size recurrent state. The states come from a real prefill of 0, 1k and 64k tokens, so history independence is measured, not assumed.

Timing uses CUDA events: warmup, 50 repeats, median with p10 and p90. Every point is also replayed from a CUDA graph. The replay median is the kernel time. Eager minus replay is the launch overhead, which matters for the short kernels. Every backend passes a numerics check against a plain PyTorch reference (GPU tests) before it is timed.

The supporting ops (FP8 and BF16 GEMMs at the model's fixed shapes, norms, activation, quantization, rotary) are timed at the token counts of a step. The step distribution under load comes from closed loop serving runs with prefix caching on and off. Every engine step carries its composition and duration from the profiler trace. Both feed a sum of parts: 48 x GDN + 16 x attention + everything else, compared to the measured step time and to the roofline.

## Deliverables

- this repo: harness, configs, raw results, processed tables, figures
- [`report/REPORT.md`](report/REPORT.md): methodology, results and limitations

## Layout

```
bench/core        timer, roofline helpers, nvidia-smi queries, environment record, ceilings, sweep runner, regimes, mock hardware
bench/census      analytic inventory and shape variants, server log parser, profiler trace parser, inventory diff, profile driver, step distribution
bench/attention   FlashAttention 3 driver, reference, accounting, runner
bench/gdn         GDN kernel drivers, reference, accounting, runner
bench/ops         supporting ops: the projections and elementwise ops at the token counts of a step
analysis          process (raw rows to CSV), crossover, perf_model (roofline model), sum_of_parts, figures
configs           model config and sweep definitions
scripts           host bootstrap, container helper, server start with the profiler
results           env, census, raw, processed
report            REPORT.md and figures
```

## Running it

Without a GPU, everything runs in mock mode with an analytic latency model:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/python -m bench.attention.run -c configs/attention_warm.yaml --mock
.venv/bin/python -m analysis.process
.venv/bin/python -m analysis.crossover
.venv/bin/python -m analysis.perf_model
.venv/bin/python -m analysis.figures
```

The analysis commands run on the development machine with the `dev` extras. The vLLM image has no matplotlib, so `analysis.figures` does not run inside the container. The committed raw rows regenerate every table and figure. `analysis.process` writes the CSVs, `analysis.crossover` the attention versus GDN crossover, `analysis.perf_model` the roofline model and kernel attainment, and `analysis.figures` the PNG and SVG figures in `report/figures`. The census tables come from `bench.census.analytic` and `bench.census.variants`. The observed side comes from `bench.census.trace_parser --trace <profiler trace> --mode eager` followed by `bench.census.diff`.

On the GPU host the benchmarks run inside the pinned vLLM image, so kernel versions match the engine:

```bash
bash scripts/bootstrap_host.sh
bash scripts/container.sh start
bash scripts/container.sh exec python3 -m bench.core.env
bash scripts/container.sh exec python3 -m bench.core.ceilings
bash scripts/container.sh exec python3 -m bench.attention.run -c configs/attention_warm.yaml
bash scripts/container.sh exec python3 -m bench.gdn.run -c configs/gdn_warm.yaml
bash scripts/container.sh exec python3 -m bench.ops.run -c configs/ops_gemm.yaml
bash scripts/container.sh exec python3 -m bench.core.floor
```

The census against the live server needs the server started with the profiler. Eager mode gives the operator shapes and the default mode gives the production kernel set. The step distribution runs a closed loop load and reads the step compositions from the profiler's step annotations:

```bash
TRACES_DIR=$HOME/traces bash scripts/serve.sh compiled
bash scripts/container.sh exec python3 -m bench.census.profile_driver --mode compiled --trace-dir /traces
bash scripts/container.sh exec python3 -m bench.census.step_distribution --caching on --trace-dir /traces --concurrency 1 --concurrency 8 --concurrency 32 --concurrency 128
bash scripts/container.sh exec python3 -m bench.census.trace_parser --mode compiled --append --trace /traces/<rank0 trace> --label <point>
bash scripts/container.sh exec python3 -m bench.census.diff
```

Sweeps are resumable. Finished points are skipped, `--only <glob>` reruns a subset and `--dry-run` lists the points. Every result row carries the git commit, the image digest and an environment hash. The committed results were produced from committed code.

The profiler traces themselves are not committed, since they are tens of MB each. The parsed tables in `results/census` are, together with the manifest of the capture grid and both server startup logs. `PREFIX_CACHING=off bash scripts/serve.sh compiled` starts the server without prefix caching for the cold runs.

## Scope

Three things were cut. The separate end to end anchor tool, because the step distribution traces carry measured step times and compositions and serve as the anchor. The 64 sequence decode point over 16k tokens in the compiled census. And the extra attention backend comparison points.

## Status

Done. The ceilings, the census from the config, from eager and compiled profiler traces and from the engine under load. The attention and GDN sweeps in all regimes, the supporting ops, the roofline model and the sum of parts against measured engine steps. The write up is [`report/REPORT.md`](report/REPORT.md).

## License

Apache-2.0
