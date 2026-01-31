from typing import Iterable, Optional

from mineru_vl_utils.structs import BlockType
from mineru_vl_utils import MinerUClient
from mineru_vl_utils import MinerULogitsProcessor
from vllm import LLM


def create_mineru_client(model_name: str = "opendatalab/MinerU2.5-2509-1.2B"):

    logits_processors = [MinerULogitsProcessor]

    # Fix for vLLM ValidationError: conflicts between 'rope_type=default' and 'type=mrope'
    # This is a known issue with some model configs in newer vLLM versions.
    llm_kwargs = {
        "model": model_name,
        "trust_remote_code": True,
        # Explicitly disable rope scaling to avoid pydantic validation conflict
        # "rope_scaling": None 
    }
    if logits_processors is not None:
        llm_kwargs["logits_processors"] = logits_processors

    llm = LLM(**llm_kwargs)
    return MinerUClient(
        backend="vllm-engine",
        vllm_llm=llm,
    )


def blocks_to_text(blocks: Optional[Iterable]) -> str:
    if not blocks:
        return ""
    text_types = {
        BlockType.TEXT,
        BlockType.TITLE,
        BlockType.LIST,
        BlockType.CODE,
        BlockType.ALGORITHM,
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.PAGE_NUMBER,
        BlockType.PAGE_FOOTNOTE,
        BlockType.ASIDE_TEXT,
        BlockType.REF_TEXT,
        BlockType.PHONETIC,
        BlockType.UNKNOWN,
        BlockType.TABLE_CAPTION,
        BlockType.IMAGE_CAPTION,
        BlockType.CODE_CAPTION,
        BlockType.TABLE_FOOTNOTE,
        BlockType.IMAGE_FOOTNOTE,
        BlockType.EQUATION,
        BlockType.EQUATION_BLOCK,
    }
    parts = []
    for block in blocks:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        content = block.get("content") if isinstance(block, dict) else getattr(block, "content", None)
        if block_type in text_types and content:
            parts.append(content)
    return " ".join(parts)
