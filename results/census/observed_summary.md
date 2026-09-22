| mode | trace | step | instances | kernel launches | distinct kernels | gdn us | attention us | mlp us | norm us | embedding us | logits us | sampler us | rotary us | kv_cache us | other us | total us |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eager | prompt_1001_decode_7 | execute_context_0(0)_generation_1(1) | 7 | 2668 | 46 | 2550 | 918 | 5397 | 2710 | 3 | 0 | 3 | 0 | 6 | 4 | 11591 |
| eager | prompt_1001_decode_7 | execute_context_1(217)_generation_0(0) | 1 | 3665 | 62 | 8454 | 1990 | 9660 | 6055 | 4 | 0 | 2 | 0 | 22 | 8 | 26196 |
| eager | prompt_1001_decode_7 | execute_context_1(784)_generation_0(0) | 1 | 3415 | 61 | 17467 | 4098 | 29638 | 11811 | 6 | 0 | 2 | 0 | 36 | 10 | 63068 |
| eager | prompt_1001_decode_7 | outside_step | 1 | 108 | 12 | 20 | 0 | 0 | 0 | 0 | 5316 | 167 | 0 | 0 | 64 | 5567 |
