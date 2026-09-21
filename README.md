# qwen27b-kernel-bench

Shape census and kernel level benchmarks for **Qwen3.8-27B** on a single **NVIDIA H200**, with vLLM 0.29 as the engine of record. I planned this for a B300, but Verda had none in stock, so the numbers come from an H200 141GB. The method is the same on any GPU because every ceiling is measured on the device.

Qwen3.8-27B is a hybrid: 48 Gated DeltaNet (GDN) layers and 16 full attention layers. This repo answers three questions about it:

1. Which operators does the model actually run, at which shapes, and how are those shapes distributed under real continuous batching?
2. How do the two mixers, full attention and GDN, behave at kernel level when the cache is **warm** (most KV already cached) and when the batch is **ragged** (many different sequence lengths in one batch)?
3. Do the measured kernel times add up to the step time the engine reports, and how large is the part they do not explain?

Everything is normalized to ceilings measured on the same GPU (achieved HBM bandwidth, achieved FP8 and BF16 GEMM throughput), not to spec sheet numbers.

## Model geometry

| | Value |
|---|---|
| Layers | 64 = 16 x (3 GDN + 1 full attention); attention at layers 3, 7, ..., 63 |
| Hidden / MLP / vocab | 5120 / 17408 / 248320 |
| Attention | 24 query heads, 4 KV heads, head_dim 256, output gate, rotary on 64 dims |
| GDN | 16 key heads, 48 value heads, head_dim 128, conv kernel 4 over 10240 channels |
| GDN state per sequence per layer | recurrent (48, 128, 128) fp32 = 3 MiB, conv (10240, 3) bf16 |
| GEMMs as vLLM runs them | GDN `in_proj_qkvz` 5120->16384, `in_proj_ba` 5120->96, `out_proj` 6144->5120; attention `qkv_proj` 5120->14336, `o_proj` 6144->5120; MLP `gate_up_proj` 5120->34816, `down_proj` 17408->5120 |
| KV cache in vLLM | manager block 784 tokens (prefix cache granularity), attention kernel page 16 tokens |

## What gets measured

**Shape census**, from three sources that are diffed against each other:

- analytic: every operator and shape derived from the model config, plus how shapes change with tensor parallel 1/2/4/8, fp8 KV cache, speculative decoding and prefix caching
- observed: torch profiler traces from a live vLLM server, once in eager mode (operator shapes) and once in the default compiled CUDA graph mode (the kernels that run in production)
- real distribution: prefill tokens, decode tokens and sequence count of every engine step under load, with prefix caching off and on

**Kernel benchmarks** of the attention and GDN paths vLLM calls (paged prefill, append and decode attention, causal conv1d, chunked gated delta rule, fused recurrent decode), in the same regimes for both mixers:

| Regime | Definition |
|---|---|
| Cold | no cached tokens; prefill over 128 to 65536 tokens, decode over KV 1k to 128k |
| Warm, fraction view | context L in multiples of 7840 tokens, cached fraction 0 / 0.5 / 0.9 / 0.99 aligned to 784 token blocks, only the new tokens are computed |
| Warm, fixed new view | 64 / 512 / 2048 new tokens over cached history from 0 to 128k |
| Ragged prefill | equal total new tokens, lengths uniform / 80 to 100% jitter / lognormal / bimodal / mixed prefill+decode |
| Ragged decode | one new token per sequence, KV lengths uniform / jitter / lognormal / bimodal at equal total KV |

For GDN, "warm" means a restored fixed size recurrent state. The states are produced by really prefilling 0, 1k and 64k tokens, so history independence is measured rather than assumed.

**Timing**: CUDA events, warmup, 50 repeats, median with p10/p90. Kernels under 100 microseconds are also replayed from a CUDA graph and both numbers are reported, since at that scale eager timing mostly measures launch overhead. Every backend passes a numerics check against a plain PyTorch reference before it is timed.

**Supporting ops** (FP8 and BF16 GEMMs at the model's fixed shapes, norms, activation, rotary) and a **thin end to end anchor** (short serving runs, cold and about 90% prefix hits) feed a sum of parts model: 48 x GDN + 16 x attention + everything else, compared to measured step time.

## Deliverables

- this repo: harness, configs, raw results, processed tables, figures
- [`report/REPORT.md`](report/REPORT.md): methodology, results, limitations, and the challenges I hit and how I solved them

## Layout

```
bench/core        timer, roofline model, nvidia-smi queries, environment record, ceilings, mock hardware
bench/census      analytic inventory, profiler driver, trace parser, step distribution, shape set
bench/attention   attention kernel drivers, reference, accounting, runner
bench/gdn         GDN kernel drivers, reference, accounting, runner
bench/ops         GEMM and elementwise ops
bench/anchor      end to end anchor runs
analysis          processing, figures, sum of parts, summary, report check
configs           model, shapes and sweep definitions
scripts           host bootstrap, container helper, overnight sweep chain
results           env, census, raw, processed, anchor
```

## Running it

Without a GPU, everything runs in mock mode with an analytic latency model:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/python -m bench.attention.run -c configs/attention_warm.yaml --mock
.venv/bin/python -m analysis.process && .venv/bin/python -m analysis.figures
```

On the GPU host the benchmarks run inside the pinned vLLM image, so kernel versions match the engine:

```bash
bash scripts/bootstrap_host.sh
bash scripts/container.sh start
bash scripts/container.sh exec python3 -m bench.core.env
bash scripts/container.sh exec python3 -m bench.core.probes
bash scripts/container.sh exec python3 -m bench.core.ceilings
bash scripts/container.sh exec python3 -m bench.attention.run -c configs/attention_warm.yaml
bash scripts/container.sh exec python3 -m bench.gdn.run -c configs/gdn_warm.yaml
```

Sweeps are resumable: finished points are skipped, `--only <glob>` reruns a subset, `--dry-run` lists the points.

## Timeline

- **Day 1**: environment and probes on the GPU, measured ceilings, analytic census, harness, attention and GDN drivers with numerics checks, sweeps launched overnight.
- **Day 2**: triage, profiler census and step distributions, supporting ops and anchor, figures, report.

If time runs short the cut order is: end to end anchor, attention backend comparison points, the bimodal and mixed ragged distributions, supporting ops. Warm attention versus warm GDN is never cut.

## Status

Work in progress, day 1.

## License

Apache-2.0
