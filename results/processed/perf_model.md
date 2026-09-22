| scenario | new tokens | sequences | embedding us | every_layer us | mlp us | attention us | gdn us | head us | total us | memory bound share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold prefill chunk 8192 | 8192 | 1 | 39 | 24982 | 205888 | 39838 | 75543 | 595 | 346884 | 16% |
| warm chunk 784 over 7056 | 784 | 1 | 4 | 2391 | 19704 | 5256 | 7308 | 595 | 35257 | 16% |
| decode batch 1 at 4k | 1 | 1 | 0 | 3 | 4006 | 456 | 1380 | 595 | 6439 | 100% |
| decode batch 64 at 4k | 64 | 64 | 0 | 195 | 4201 | 4448 | 6054 | 602 | 15501 | 100% |
| decode batch 64 at 64k | 64 | 64 | 0 | 195 | 4201 | 64705 | 6054 | 602 | 75758 | 100% |
| decode batch 256 at 16k | 256 | 256 | 1 | 781 | 6434 | 64971 | 20801 | 910 | 93898 | 90% |
| mixed: 1 chunk 4096 + 32 decodes at 16k | 4128 | 33 | 20 | 12588 | 103748 | 19265 | 40394 | 599 | 176614 | 22% |

| benchmark | backend | points | median attainment | min | max |
|---|---|---:|---:|---:|---:|
| attention_cold | flash_attn_3 | 36 | 84% | 6% | 101% |
| attention_pages | flash_attn_3 | 28 | 78% | 30% | 95% |
| attention_ragged | flash_attn_3 | 42 | 71% | 22% | 101% |
| attention_warm | flash_attn_3 | 121 | 84% | 3% | 89% |
| gdn_cold | flashinfer | 15 | 17% | 4% | 24% |
| gdn_cold | triton | 15 | 8% | 4% | 9% |
| gdn_decode | packed | 7 | 85% | 15% | 93% |
| gdn_decode | sigmoid_gating | 7 | 77% | 16% | 91% |
| gdn_ragged | flashinfer | 30 | 20% | 14% | 25% |
| gdn_ragged | triton | 30 | 9% | 8% | 10% |
| gdn_warm | flashinfer | 58 | 20% | 4% | 25% |
| gdn_warm | triton | 58 | 8% | 4% | 14% |
