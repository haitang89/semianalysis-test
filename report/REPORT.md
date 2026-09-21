# Attention and Gated DeltaNet kernels of Qwen3.8-27B on an H200: shapes, warm caches and ragged batches

Work in progress. The methodology is written, results come on day 2.

## 1. Methodology

### 1.1 Hardware and software

| Item | Value |
|---|---|
| GPU | 1x NVIDIA H200 141GB on Verda, compute capability 9.0, 132 SMs, driver 580.178.04, 700 W power limit |
| Engine | vLLM 0.29.0, CUDA 13 image, digest recorded in `results/env/environment.json` |
| Kernel libraries | the FlashInfer, Triton and vendored linear attention kernels shipped inside that image |
| Model | `Qwen/Qwen3.8-27B-FP8`, text path only (`--language-model-only`) |
| Where benchmarks run | inside the same container as the engine, so kernel versions are identical |

### 1.2 Ceilings

All utilization numbers are relative to ceilings measured on this GPU: device memory bandwidth from copy and reduction kernels at 1, 4 and 16 GiB, and BF16 and FP8 GEMM throughput at large model shaped problems. Vendor figures are listed beside them for reference only. Clocks and power are sampled during the ceiling runs and throttling is flagged.

### 1.3 Shape census

Three sources, diffed against each other:

1. **Analytic** inventory from the model config, following vLLM's fused projection layout, with per step counts (48 GDN layers, 16 attention layers, 64 MLPs) and a table of shape variants under tensor parallel 1/2/4/8, fp8 KV, speculative decoding and prefix caching. Variants other than the benchmarked configuration are labelled analytic only.
2. **Observed** torch profiler traces from the live server over a prefill and decode grid, in eager mode (operator shapes visible) and in the default compiled CUDA graph mode (production kernel set, padded token counts).
3. **Real distribution** of prefill tokens, decode tokens and sequence count per engine step at concurrency 1, 8, 32 and 128, with prefix caching off and on.

The benchmark shape set is derived from these (grid plus observed percentiles); each shape records its source.

### 1.4 Regimes

| Regime | Definition |
|---|---|
| Cold | no cached tokens |
| Warm, fraction view | context L in multiples of 7840 tokens; cached fraction 0, 0.5, 0.9, 0.99 aligned to the 784 token manager block; only new tokens are computed over the cached KV |
| Warm, fixed new view | 64, 512, 2048 new tokens over cached history 0 to 128k |
| Ragged prefill | fixed total new tokens (4096, 16384) over 8, 32, 128 sequences; lengths uniform, 80 to 100% jitter, lognormal (sigma 1.0), bimodal, mixed prefill+decode; fixed seeds |
| Ragged decode | one new token per sequence; KV lengths uniform, jitter, lognormal, bimodal at equal total KV tokens |

The attention kernel page is 16 tokens, as vLLM configures it for this model; 32, 64 and 128 are measured as a comparison. For GDN the warm case passes a restored recurrent and conv state; the states come from really prefilling 0, 1k and 64k tokens.

### 1.5 Timing and correctness

CUDA events, 10 warmup calls, 50 timed repeats, one synchronize per repeat batch; median with p10 and p90. A point whose spread exceeds 25% of the median is measured again once with double repeats and flagged if still unstable. Kernels under 100 microseconds are also captured into a CUDA graph and replayed; both medians and their difference (launch overhead) are reported. Before timing, every backend is compared to a plain PyTorch reference at a small shape; for GDN, a final state fed back as the initial state must reproduce the uninterrupted result. Backends that fail are still recorded but marked.

### 1.6 Derived metrics

Causal aware FLOPs and bytes moved per call give achieved TFLOPS and GB/s, utilization against the measured ceilings, microseconds per new token, and per step cost (x16 for attention, x48 for GDN). `ragged_efficiency` is the uniform batch time divided by the ragged batch time at equal total work. The crossover is the cached history length where per layer attention time exceeds per layer GDN time at the same new token count.

### 1.7 Sum of parts

For each measured engine step composition, predicted time = 48 x GDN + 16 x attention + supporting ops at the nearest benchmarked shapes. The difference to the measured step time is reported as the unexplained residual. This sum is a bound, not a model of the engine: kernel overlap and framework work are not represented.

## 2. Shape census

## 3. Ceilings

## 4. Attention results

## 5. GDN results

## 6. Attention versus GDN

## 7. Supporting ops

## 8. Sum of parts against measured step time

## 9. Limitations and future work

## 10. Challenges

