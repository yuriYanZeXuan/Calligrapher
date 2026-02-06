"""
MinerU OCR 工具（transformers backend）。

按你提供的参考代码重构：
- 彻底移除 vLLM 相关逻辑
- 仅保留对外接口：`create_mineru_client` / `blocks_to_text`
"""

from __future__ import annotations

from typing import Iterable, Optional, Any


def create_mineru_client(model_name: str = "opendatalab/MinerU2.5-2509-1.2B"):
    """
    创建 MinerU OCR client（transformers backend）。

    Args:
        model_name: HuggingFace 模型名或本地路径
    """
    # 延迟导入：避免 import 本模块时因环境依赖缺失直接报错
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration  # type: ignore
    from mineru_vl_utils import MinerUClient  # type: ignore

    # 使用 torch_dtype 参数
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        use_fast=True,
    )
    # 简单粗暴修复：如果 config 缺 max_position_embeddings，直接补一个默认值
    if not hasattr(model.config, "max_position_embeddings"):
        model.config.max_position_embeddings = 32768
    return MinerUClient(
        backend="transformers",
        model=model,
        processor=processor,
    )


def blocks_to_text(blocks: Optional[Iterable]) -> str:
    """把 MinerU 输出 blocks 提取成纯文本（用于 OCR 指标）。"""
    from mineru_vl_utils.structs import BlockType  # type: ignore

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

def blocks_to_text_with_bbox(blocks: Optional[Iterable]) -> str:
    """
    把 MinerU 输出 blocks 提取成纯文本，并返回每个字段的 bbox 和旋转信息。
    
    Args:
        blocks: MinerU 输出的 blocks 列表
        
    Returns:
        OCRResult: 包含以下字段:
            - text: 合并后的纯文本（用于 OCR 指标）
            - blocks: 每个识别块的详细信息列表，每项包含:
                - type: 块类型
                - content: 文本内容
                - bbox: 边界框 [xmin, ymin, xmax, ymax]，归一化到 0-1
                - angle: 旋转角度 (None, 0, 90, 180, 270)
    """
    from mineru_vl_utils.structs import BlockType  # type: ignore

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
    
    text_parts = []
    block_infos: list[dict[str, Any]] = []
    
    for block in blocks:
        # 兼容 dict 和对象两种访问方式
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        content = block.get("content") if isinstance(block, dict) else getattr(block, "content", None)
        bbox = block.get("bbox") if isinstance(block, dict) else getattr(block, "bbox", None)
        angle = block.get("angle") if isinstance(block, dict) else getattr(block, "angle", None)
        
        if block_type in text_types and content:
            text_parts.append(content)
            block_infos.append({
                "type": block_type,
                "content": content,
                "bbox": bbox if bbox else [0.0, 0.0, 1.0, 1.0],  # 默认全图
                "angle": angle,  # None, 0, 90, 180, 270
            })
    
    return {
        "text": text_parts,
        "blocks": block_infos,
    }


def main():
    """调试 MinerU 功能的主函数"""
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="调试 MinerU OCR 功能")
    parser.add_argument("--image_path", type=str, help="输入图片路径")
    parser.add_argument("--model", type=str, default="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/MinerU_VLM", help="模型名称")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")
    args = parser.parse_args()
    
    # 检查图片是否存在
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"错误: 图片文件不存在: {image_path}")
        return
    
    print(f"加载模型: {args.model}")
    print(f"使用设备: {args.device}")
    
    # 初始化客户端
    client = create_mineru_client(model_name=args.model)
    
    print(f"\n处理图片: {image_path}")
    
    # 执行 OCR
    from PIL import Image
    image = Image.open(str(image_path))
    result = client.two_step_extract(image)
    
    # 提取文本
    text = blocks_to_text(result)
    text_with_bbox = blocks_to_text_with_bbox(result)
    print("\n" + "="*50)
    print("OCR 结果:")
    print("="*50)
    print(text)
    print("="*50)
    
    # 打印详细的 blocks 信息
    print("\n" + "="*50)
    print("OCR 结果 (含 bbox 和旋转):")
    print("="*50)
    print(text_with_bbox)
    print("="*50)

if __name__ == "__main__":
    main()