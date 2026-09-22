| block | op | kind | dtype | shape | per step | weight MiB |
|---|---|---|---|---|---:|---:|
| embedding | embed_tokens | gather | bf16 | [M] -> [M, 5120] | 1 | 2425.0 |
| every_layer | input_layernorm | rmsnorm | bf16 | [M, 5120] | 64 | 0.0 |
| every_layer | post_attention_layernorm | rmsnorm | bf16 | [M, 5120] | 64 | 0.0 |
| every_layer | residual_add | elementwise | bf16 | [M, 5120] x2 | 128 |  |
| mlp | gate_up_proj | gemm | fp8 | [M, 5120] x [5120, 34816] -> [M, 34816] | 64 | 170.0 |
| mlp | silu_and_mul | elementwise | bf16 | [M, 34816] -> [M, 17408] | 64 |  |
| mlp | down_proj | gemm | fp8 | [M, 17408] x [17408, 5120] -> [M, 5120] | 64 | 85.0 |
| attention | qkv_proj | gemm | fp8 | [M, 5120] x [5120, 14336] -> [M, 14336] | 16 | 70.0 |
| attention | q_norm | rmsnorm | bf16 | [M, 24, 256] | 16 | 0.0 |
| attention | k_norm | rmsnorm | bf16 | [M, 4, 256] | 16 | 0.0 |
| attention | rotary | elementwise | bf16 | [M, 28, 64 of 256] | 16 |  |
| attention | kv_cache_write | scatter | bf16 | [M, 2, 4, 256] -> paged cache | 16 |  |
| attention | attention | attention | bf16 | q [M, 24, 256], kv [L, 4, 256] per sequence | 16 |  |
| attention | output_gate | elementwise | bf16 | [M, 6144] * sigmoid([M, 6144]) | 16 |  |
| attention | o_proj | gemm | fp8 | [M, 6144] x [6144, 5120] -> [M, 5120] | 16 | 30.0 |
| gdn | in_proj_qkvz | gemm | fp8 | [M, 5120] x [5120, 16384] -> [M, 16384] | 48 | 80.0 |
| gdn | in_proj_ba | gemm | bf16 | [M, 5120] x [5120, 96] -> [M, 96] | 48 | 0.9 |
| gdn | causal_conv1d | conv | bf16 | [B, 10240, T] kernel 4, state [B, 10240, 3] | 48 | 0.1 |
| gdn | post_conv_prep | elementwise | bf16 | split [M, 10240] -> q,k [M, 16, 128], v [M, 48, 128]; l2norm; gates from a, b | 48 | 0.0 |
| gdn | gated_delta_rule | linear_attention | bf16 | q,k [M, 16, 128], v [M, 48, 128], state [B, 48, 128, 128] fp32 | 48 |  |
| gdn | gated_rmsnorm | rmsnorm | bf16 | [M, 6144] gated by z | 48 | 0.0 |
| gdn | state_gather | gather | fp32 | [B, 48, 128, 128] from the state pool | 48 |  |
| gdn | state_checkpoint | copy | fp32 | recurrent and conv states copied at block boundaries | 1 |  |
| gdn | out_proj | gemm | fp8 | [M, 6144] x [6144, 5120] -> [M, 5120] | 48 | 30.0 |
| head | final_norm | rmsnorm | bf16 | [M, 5120] | 1 | 0.0 |
| head | lm_head | gemm | bf16 | [M, 5120] x [5120, 248320] -> [M, 248320] | 1 | 2425.0 |
| every_layer | fp8_activation_quant | quantize | fp8 | [M, K] bf16 -> fp8 with per token group scales, before each FP8 GEMM | 256 |  |
| mtp | mtp_fc | gemm | bf16 | [M, 10240] x [10240, 5120] -> [M, 5120] | 0 | 100.0 |
