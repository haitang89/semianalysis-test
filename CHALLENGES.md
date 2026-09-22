# Challenges

What went wrong along the way and what I did about it. Short notes, roughly in order.

### The H200 loaded other kernels

I had a list of kernels from reading vLLM's code with a Blackwell GPU in mind. The first server start on the H200 showed a different set: FlashAttention 3 for attention, FlashInfer for GDN prefill, a Triton kernel for GDN decode, DeepGEMM for the FP8 GEMMs. The page size changed with it too, since FlashAttention runs on vLLM's 784 token block as is. The log was not the whole truth either. It names a CUDA kernel for GDN decode that the source only uses for speculative decoding, and the eager trace showed the Triton kernel instead. The GEMM shapes were not the checkpoint's shapes, because vLLM fuses the four GDN projections into two. I threw the list away and read everything from the startup log, the source and the profiler trace.

### "Throttled" was the power cap

My first ceilings run flagged the GPU as throttled because the SM clock sat well below its maximum. It was not a fault. A sustained GEMM pulls 693 W against a 700 W limit and the driver lowers the clock to stay under it. I changed the check to read the driver's own slowdown reasons. Power capping is now reported separately, because it is the normal state under load and every sustained number is taken in it.

### Calling the GDN kernels outside the engine

Two traps. The conv kernel treats state slot 0 as a null block and silently skips any sequence that points at it, so my first numerics check came back all zeros. And the recurrent state is stored transposed, as heads by value dim by key dim. My reference used the textbook layout, so outputs matched but final states did not. Slot 1 and a transposed reference fixed both. After that every kernel matches the reference within bf16 tolerance, including a state fed back in.

### The chunked GDN kernels crash on huge calls

Both chunked prefill kernels died with an illegal memory access once a single call went past a few hundred thousand tokens. It looks like 32 bit indexing. The engine never gets there, because a prefill step is capped at 8192 tokens. I cap a call at 131072 tokens in the driver and record anything larger as unsupported.

### Warm GDN can only be measured at kernel level

vLLM saves GDN state only at 784 token block boundaries, so a server run cannot show fine grained state reuse. I measure the warm case by prefilling a real history, saving the state, and passing it back in. That also let me check that the cost does not depend on how long the history was. It does not.

### Some warm attention points took too long

Seven warm attention points had calls of several seconds, five at batch 32 over 62k and 125k contexts and two at batch 8. 60 calls do not fit the point timeout. Two were recorded as timeouts. In the other five the timeout fired inside the CUDA graph capture. The capture fallback caught it and the point was recorded as fine with its eager median. They are valid eager numbers, so I kept them and marked them. The timeout is now an exception nothing can swallow, and the timer synchronizes per call once a call is longer than 100 ms.

### Profiling the server

Three things here. The profiler flags I knew from older vLLM were gone, and shape recording is off by default. The default compiled mode hides shapes, so I profile twice: eager for the shapes, compiled for the kernels that really run. And stopping the profiler blocks the whole server until the trace is written. For a 64 sequence decode over 16k tokens the export never seemed to end, so I dropped that one point.

### Reading the compiled trace

Kernels replayed from a CUDA graph carry no operator and no Python frames. My first parse labelled almost everything as attention, because the only frame it could see was the decoder layer. I turned the labelling around: kernel name and operator name decide first, frames are a fallback. GEMM shapes come from the template arguments in the DeepGEMM kernel names. One shape is shared by two projections, and I split it between them by layer count.

### The trace corrected my inventory

The diff of observed kernels against my analytic inventory found 256 activation quantization kernels per step where I had 304. I had counted one BF16 projection as FP8. It also found two ops I had not listed, the gather of GDN states before the chunked kernel and the state copy at block boundaries. Both are in the inventory now and nothing in the traces is left unexplained.

### The kernels did not add up to the step

My first sum of parts came out 39 percent above the measured decode step. The cause was my own timing. Every point is a CUDA graph replay of one kernel, and a replay costs about 8 us before the kernel does anything. A decode step is about 1100 kernels, and the engine replays the whole step as one graph, so it pays that cost once. I measured that floor. Taking all of it off every kernel overshoots and the sum lands below the step. So I calibrated the correction against the compiled trace instead. At 4 us per kernel the isolated GEMMs match their time inside the engine's graph. With that the kernel sum lands within 6 percent of the measured step at every concurrency.
