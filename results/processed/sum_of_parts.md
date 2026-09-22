| mode | caching | concurrency | steps | measured us p50 | kernel sum us p50 | roofline us p50 | residual share p50 | attention share of kernel sum | GDN share | ops share |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compiled | off | 1 | 538 | 9582 | 10154 | 6393 | -6% | 2% | 3% | 95% |
| compiled | off | 8 | 398 | 10819 | 10926 | 7079 | -1% | 2% | 9% | 90% |
| compiled | off | 32 | 131 | 13927 | 13444 | 9429 | 3% | 2% | 12% | 85% |
| compiled | off | 128 | 99 | 24882 | 25295 | 18829 | -2% | 7% | 28% | 66% |
| compiled | on | 1 | 531 | 9673 | 10154 | 6393 | -5% | 2% | 3% | 95% |
| compiled | on | 8 | 383 | 10898 | 10926 | 7079 | -0% | 2% | 10% | 89% |
| compiled | on | 32 | 257 | 14050 | 13444 | 9429 | 4% | 3% | 15% | 81% |
| compiled | on | 128 | 75 | 24448 | 25295 | 18829 | -3% | 5% | 20% | 75% |
