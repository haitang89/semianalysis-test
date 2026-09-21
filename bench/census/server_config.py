"""What the engine decided at startup, read from its own log rather than assumed."""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

PATTERNS = {
    "attention_backend": re.compile(r"Using (\w+) attention backend"),
    "attention_candidates": re.compile(r"out of potential backends: \[([^\]]*)\]"),
    "flash_attention_version": re.compile(r"Using FlashAttention version (\d+)"),
    "gdn_prefill_kernel": re.compile(r"Using (\w+) GDN prefill kernel"),
    "gdn_decode_kernel": re.compile(r"GDN decode kernel: (\w+)"),
    "fp8_gemm_kernel": re.compile(r"Selected (\w+) for Fp8LinearMethod"),
    "attention_block_tokens": re.compile(r"Setting attention block size to (\d+) tokens"),
    "kernel_block_tokens": re.compile(r"Setting kv cache block size to (\d+) for"),
    "mamba_cache_mode": re.compile(r"Mamba cache mode is set to '(\w+)'"),
    "max_num_batched_tokens": re.compile(r"max_num_batched_tokens=(\d+)"),
    "kv_cache_tokens": re.compile(r"GPU KV cache size: ([\d,]+) tokens"),
    "cudagraph_mode": re.compile(r"cudagraph_mode[=: ]+['\"]?(\w+)|Cudagraph is (disabled)"),
}
INTEGER_FIELDS = {
    "flash_attention_version",
    "attention_block_tokens",
    "kernel_block_tokens",
    "max_num_batched_tokens",
    "kv_cache_tokens",
}


@dataclass
class ServerConfig:
    attention_backend: Optional[str] = None
    attention_candidates: Optional[list] = None
    flash_attention_version: Optional[int] = None
    gdn_prefill_kernel: Optional[str] = None
    gdn_decode_kernel: Optional[str] = None
    fp8_gemm_kernel: Optional[str] = None
    attention_block_tokens: Optional[int] = None
    kernel_block_tokens: Optional[int] = None
    mamba_cache_mode: Optional[str] = None
    max_num_batched_tokens: Optional[int] = None
    kv_cache_tokens: Optional[int] = None
    cudagraph_mode: Optional[str] = None

    def to_dict(self) -> dict:
        record = asdict(self)
        record["kernel_page_tokens"] = self.kernel_page_tokens
        return record

    @property
    def kernel_page_tokens(self) -> Optional[int]:
        """Page size the attention kernel really runs on.

        FlashAttention takes any block that is a multiple of 16, so it gets the
        manager block itself. FlashInfer logs the smaller page it falls back to.
        """
        if self.attention_backend == "FLASH_ATTN" and self.attention_block_tokens:
            return self.attention_block_tokens if self.attention_block_tokens % 16 == 0 else None
        return self.kernel_block_tokens

    def missing(self) -> list[str]:
        return [name for name, value in asdict(self).items() if value is None and name != "kernel_block_tokens"]


def _first_group(match: re.Match) -> str:
    return next(group for group in match.groups() if group is not None)


def parse_server_log(lines: Iterable[str]) -> ServerConfig:
    config = ServerConfig()
    for line in lines:
        for field, pattern in PATTERNS.items():
            if getattr(config, field) is not None:
                continue
            match = pattern.search(line)
            if not match:
                continue
            value = _first_group(match)
            if field == "attention_candidates":
                setattr(config, field, re.findall(r"\w+", value))
            elif field in INTEGER_FIELDS:
                setattr(config, field, int(value.replace(",", "")))
            else:
                setattr(config, field, value.lower() if field == "cudagraph_mode" else value)
    return config


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Parse the engine startup log into a config record")
    parser.add_argument("log", help="server log file, or - for stdin")
    parser.add_argument("--out", default="results/census/server_config.json")
    args = parser.parse_args(argv)

    lines = sys.stdin if args.log == "-" else Path(args.log).read_text(encoding="utf-8").splitlines()
    config = parse_server_log(lines)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config.to_dict()))
    if config.missing():
        print(f"not found in log: {', '.join(config.missing())}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
