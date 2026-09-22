| mode | caching | concurrency | steps | pure decode | mixed | pure prefill | tokens per step p50 / p90 / p99 | prefill tokens per step p50 / p90 / max | decode sequences p50 / max | prefill chunks at a 784 multiple | step us p50 |
|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|
| compiled | off | 1 | 538 | 534 | 0 | 4 | 1 / 1 / 1 | 627 / 723 / 723 | 1 / 1 | 0 of 4 | 9582 |
| compiled | off | 8 | 398 | 389 | 6 | 3 | 8 / 8 / 2435 | 2428 / 4299 / 7111 | 8 / 8 | 0 of 9 | 10819 |
| compiled | off | 32 | 131 | 123 | 8 | 0 | 32 / 32 / 8192 | 7946 / 8174 / 8182 | 32 / 32 | 0 of 8 | 13927 |
| compiled | off | 128 | 99 | 95 | 4 | 0 | 128 / 128 / 8192 | 8087 / 8094 / 8094 | 128 / 128 | 0 of 4 | 24882 |
| compiled | on | 1 | 531 | 527 | 0 | 4 | 1 / 1 / 1 | 627 / 723 / 723 | 1 / 1 | 0 of 4 | 9673 |
| compiled | on | 8 | 383 | 372 | 11 | 0 | 8 / 8 / 1161 | 1069 / 1992 / 3879 | 8 / 8 | 1 of 11 | 10898 |
| compiled | on | 32 | 257 | 247 | 10 | 0 | 32 / 32 / 3280 | 2691 / 6842 / 7738 | 32 / 32 | 1 of 10 | 14050 |
| compiled | on | 128 | 75 | 68 | 7 | 0 | 128 / 128 / 8052 | 7849 / 7946 / 8086 | 128 / 128 | 0 of 7 | 24448 |
