import json

from bench.census import server_config as sc
from bench.census.server_config import parse_server_log

H200_LOG = """
INFO 09-21 23:17:26 [scheduler.py:277] Chunked prefill is enabled with max_num_batched_tokens=8192.
INFO 09-21 23:17:26 [config.py:625] Mamba cache mode is set to 'align' for Qwen3_5ForConditionalGeneration by default when prefix caching is enabled
INFO 09-21 23:17:26 [vllm.py:1585] Cudagraph is disabled under eager mode
INFO 09-21 23:17:40 [__init__.py:695] Selected FlashInferFp8DeepGEMMDynamicBlockScaledKernel for Fp8LinearMethod
INFO 09-21 23:17:40 [qwen_gdn_linear_attn.py:167] Using FlashInfer GDN prefill kernel (requested=auto, head_k_dim=128).
INFO 09-21 23:17:40 [qwen_gdn_linear_attn.py:519] GDN decode kernel: cuda
INFO 09-21 23:17:41 [cuda.py:492] Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION'].
INFO 09-21 23:17:41 [flash_attn.py:897] Using FlashAttention version 3
INFO 09-21 23:17:51 [interface.py:918] Setting attention block size to 784 tokens to ensure that attention page size is >= mamba page size.
INFO 09-21 23:18:14 [kv_cache_utils.py:2032] GPU KV cache size: 1,400,832 tokens, Maximum concurrency for 32,768 tokens per request: 42.75x
""".strip().splitlines()


def test_parses_every_decision_from_the_real_h200_log():
    config = parse_server_log(H200_LOG)
    assert config.attention_backend == "FLASH_ATTN"
    assert config.attention_candidates == ["FLASH_ATTN", "FLASHINFER", "TRITON_ATTN", "FLEX_ATTENTION"]
    assert config.flash_attention_version == 3
    assert config.gdn_prefill_kernel == "FlashInfer"
    assert config.gdn_decode_kernel == "cuda"
    assert config.fp8_gemm_kernel == "FlashInferFp8DeepGEMMDynamicBlockScaledKernel"
    assert config.attention_block_tokens == 784
    assert config.mamba_cache_mode == "align"
    assert config.max_num_batched_tokens == 8192
    assert config.kv_cache_tokens == 1_400_832
    assert config.cudagraph_mode == "disabled"


def test_flash_attention_runs_on_the_manager_block_itself():
    config = parse_server_log(H200_LOG)
    assert config.kernel_block_tokens is None
    assert config.kernel_page_tokens == 784
    assert config.missing() == []
    assert config.to_dict()["kernel_page_tokens"] == 784


def test_flashinfer_reports_its_own_smaller_page():
    lines = [line.replace("FLASH_ATTN attention backend", "FLASHINFER attention backend") for line in H200_LOG]
    config = parse_server_log(lines + ["Setting kv cache block size to 16 for FLASHINFER backend"])
    assert config.attention_backend == "FLASHINFER"
    assert config.kernel_block_tokens == 16 and config.kernel_page_tokens == 16


def test_first_occurrence_wins():
    lines = ["GDN decode kernel: cuda", "GDN decode kernel: triton"]
    assert parse_server_log(lines).gdn_decode_kernel == "cuda"


def test_cli_writes_json(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("\n".join(H200_LOG), encoding="utf-8")
    out = tmp_path / "server_config.json"
    assert sc.main([str(log), "--out", str(out)]) == 0
    record = json.loads(out.read_text())
    assert record["attention_block_tokens"] == 784 and record["gdn_decode_kernel"] == "cuda"
