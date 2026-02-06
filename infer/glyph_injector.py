"""
Glyph Injector: 文字渲染和 latent 注入接口

实现流程：
1. 使用系统字体白底黑字渲染文字
2. 使用大津法二值化提取文字 mask
3. 将文字模板做 flow matching inversion，保存 latent 列表
4. 在去噪过程中注入对应 timestep 的 latent
"""

import os
import math
from typing import Optional, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def get_available_font(size: int = 100) -> ImageFont.FreeTypeFont:
    """获取系统中可用的字体"""
    possible_fonts = [
        # macOS
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for font_path in possible_fonts:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
    # 如果都失败，使用默认字体
    print("警告：使用默认字体")
    return ImageFont.load_default()


def calculate_font_size(text: str, bbox_width: int, bbox_height: int) -> int:
    """计算能填满 bbox 的字体大小"""
    # 初始估算：按高度计算
    estimated_size = int(bbox_height * 0.8)
    
    # 创建测试字体
    font = get_available_font(estimated_size)
    
    # 测试文字宽度
    test_img = Image.new("RGB", (bbox_width * 2, bbox_height * 2), "white")
    draw = ImageDraw.Draw(test_img)
    
    # 获取文字边界
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # 按宽高比例调整
    scale_w = bbox_width / max(text_width, 1)
    scale_h = bbox_height / max(text_height, 1)
    scale = min(scale_w, scale_h) * 0.9  # 留一点边距
    
    return max(int(estimated_size * scale), 12)


@dataclass
class TextRegion:
    """文字区域定义"""
    bbox: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max) 归一化坐标
    content: str
    
    def to_pixel_bbox(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """转换为像素坐标"""
        x1 = int(self.bbox[0] * width)
        y1 = int(self.bbox[1] * height)
        x2 = int(self.bbox[2] * width)
        y2 = int(self.bbox[3] * height)
        return (x1, y1, x2, y2)


@dataclass
class InjectionConfig:
    """注入强度配置
    
    Attributes:
        mask_strength: 空间混合强度 (0-1)，控制文字 latent 在 mask 区域的混合权重
        timestep_ratio: 时间步注入比例 (0-1)，仅在前 X% 的去噪步骤中注入
    """
    mask_strength: float = 0.8
    timestep_ratio: float = 1.0

    def should_inject(self, step_idx: int, total_steps: int) -> bool:
        """判断当前步是否需要注入"""
        if self.timestep_ratio >= 1.0:
            return True
        return step_idx < int(total_steps * self.timestep_ratio)


class GlyphInjector:
    """
    文字注入器
    
    实现文字模板渲染、mask 提取、latent inversion 和去噪过程中的 latent 注入
    """
    
    def __init__(
        self, 
        vae,
        scheduler,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        logger=None,
    ):
        """
        初始化
        
        Args:
            vae: VAE 模型，用于 encode/decode
            scheduler: Flow matching 调度器
            device: 设备
            dtype: 数据类型
            logger: TTSLogger 实例（可选）
        """
        self.vae = vae
        self.scheduler = scheduler
        self.device = device
        self.dtype = dtype
        self.logger = logger
        
        # VAE 缩放因子
        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1) if hasattr(vae, 'config') else 8
        
    def render_text_template(
        self, 
        text: str, 
        width: int, 
        height: int,
        background_color: str = "white",
        text_color: str = "black"
    ) -> Image.Image:
        """
        渲染文字模板图像
        
        Args:
            text: 文字内容
            width: 图像宽度
            height: 图像高度
            background_color: 背景色
            text_color: 文字色
            
        Returns:
            渲染后的 PIL Image
        """
        img = Image.new("RGB", (width, height), background_color)
        draw = ImageDraw.Draw(img)
        
        # 计算合适的字体大小
        font_size = calculate_font_size(text, width, height)
        font = get_available_font(font_size)
        
        # 获取文字边界并居中
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        draw.text((x, y), text, fill=text_color, font=font)
        
        return img
    
    def extract_text_mask(self, image: np.ndarray) -> np.ndarray:
        """
        使用大津法提取文字 mask
        
        Args:
            image: BGR 图像数组
            
        Returns:
            二值化 mask，文字区域为 255
        """
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 判断文字颜色
        mean_val = np.mean(gray)
        
        if mean_val < 127:
            # 背景偏暗，文字是白色
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            white_ratio = np.sum(binary == 255) / binary.size
            text_mask = binary if white_ratio < 0.5 else cv2.bitwise_not(binary)
        else:
            # 背景偏亮，文字是黑色
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            black_ratio = np.sum(binary == 255) / binary.size
            text_mask = binary if black_ratio < 0.5 else cv2.bitwise_not(binary)
        
        # 形态学操作去噪
        kernel1 = np.ones((3, 3), np.uint8)
        kernel2 = np.ones((2, 2), np.uint8)
        text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel1)
        text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, kernel2)
        
        return text_mask
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """将 PIL Image 编码为 latent"""
        # 转换为 tensor
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor * 2.0 - 1.0  # 归一化到 [-1, 1]
        img_tensor = img_tensor.to(device=self.device, dtype=self.dtype)
        
        # VAE encode
        with torch.no_grad():
            latent = self.vae.encode(img_tensor).latent_dist.sample()
            latent = (latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            
        return latent
    
    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        """将 latent 解码为 PIL Image"""
        with torch.no_grad():
            latent = (latent / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latent, return_dict=False)[0]
            
        # 转换为 PIL Image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
        
        return Image.fromarray(image)
    
    def compute_inversion_latents(
        self, 
        latent_0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        计算 flow matching inversion 的 latent 列表
        
        Flow matching: z_t = (1 - sigma) * z_0 + sigma * noise
        
        Args:
            latent_0: 原始 latent (z_0)
            noise: 噪声
            timesteps: 时间步列表
            
        Returns:
            每个时间步对应的 latent 列表
        """
        latent_list = []
        for t in timesteps:
            # 计算 sigma (归一化时间步)
            sigma = t.float() / 1000.0
            
            # Flow matching inversion
            z_t = (1 - sigma) * latent_0 + sigma * noise
            latent_list.append(z_t.clone())
        
        # 追加 sigma=0 的 clean latent，供后置注入最后一步使用
        latent_list.append(latent_0.clone())
            
        return latent_list
    
    def prepare_injection(
        self,
        text_regions: list[TextRegion],
        image_size: Tuple[int, int],
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> dict:
        """
        准备文字注入所需的数据
        
        Args:
            text_regions: 文字区域列表
            image_size: 图像尺寸 (width, height)
            noise: 初始噪声
            timesteps: 时间步列表
            
        Returns:
            包含 mask 和 latent 列表的字典
        """
        width, height = image_size
        
        # 创建完整图像的 mask
        full_mask = np.zeros((height, width), dtype=np.uint8)
        
        injection_data = {
            "masks": [],
            "latent_lists": [],
            "regions": []
        }
        
        for region in text_regions:
            # 获取像素坐标
            x1, y1, x2, y2 = region.to_pixel_bbox(width, height)
            region_width = x2 - x1
            region_height = y2 - y1
            
            # 渲染文字模板
            text_img = self.render_text_template(
                region.content,
                region_width,
                region_height
            )
            
            # 提取 mask
            text_array = np.array(text_img)
            text_mask = self.extract_text_mask(text_array)
            
            # 将 mask 放入完整图像
            full_mask[y1:y2, x1:x2] = text_mask
            
            # 编码文字模板为 latent
            # 需要调整大小以匹配完整图像
            full_text_img = Image.new("RGB", (width, height), "white")
            full_text_img.paste(text_img, (x1, y1))
            text_latent = self.encode_image(full_text_img)
            
            # 计算 inversion latents
            latent_list = self.compute_inversion_latents(
                text_latent, noise, timesteps
            )
            
            injection_data["latent_lists"].append(latent_list)
            injection_data["regions"].append((x1, y1, x2, y2))
            
            # 日志可视化：保存渲染的文字模板、mask 和全图
            if self.logger is not None:
                ts_str = ",".join(f"{t:.1f}" for t in timesteps[:5].tolist())
                if len(timesteps) > 5:
                    ts_str += f"...({len(timesteps)} steps)"
                
                caption = (
                    f"text=\"{region.content}\"  "
                    f"bbox=({x1},{y1},{x2},{y2})  "
                    f"size={region_width}x{region_height}\n"
                    f"timesteps=[{ts_str}]  "
                    f"latent={list(text_latent.shape)}"
                )
                
                region_idx = len(injection_data["regions"]) - 1
                self.logger.save_image(text_img, f"glyph_region_{region_idx}_text",
                                       caption=caption, subfolder="glyph")
                self.logger.save_image(full_text_img, f"glyph_region_{region_idx}_full",
                                       caption=caption, subfolder="glyph")
                
                # 保存 mask 为灰度图
                mask_pil = Image.fromarray(text_mask)
                self.logger.save_image(mask_pil.convert("RGB"), f"glyph_region_{region_idx}_mask",
                                       caption=caption, subfolder="glyph")
        
        # 将 mask 下采样到 latent 空间
        # ZImage/Flux latent 尺寸 = 2 * (pixel / (vae_scale_factor * 2))
        latent_h = 2 * (height // (self.vae_scale_factor * 2))
        latent_w = 2 * (width // (self.vae_scale_factor * 2))
        mask_latent = cv2.resize(full_mask, (latent_w, latent_h), interpolation=cv2.INTER_NEAREST)
        mask_latent = torch.from_numpy(mask_latent).float() / 255.0
        mask_latent = mask_latent.unsqueeze(0).unsqueeze(0).to(self.device)
        
        injection_data["mask_latent"] = mask_latent
        injection_data["full_mask"] = full_mask
        injection_data["total_steps"] = len(timesteps)
        
        return injection_data
    
    def inject_latent(
        self,
        current_latent: torch.Tensor,
        injection_data: dict,
        step_idx: int,
        config: InjectionConfig = None
    ) -> torch.Tensor:
        """
        在当前 latent 中注入文字区域的 latent（后置注入）
        
        Args:
            current_latent: scheduler step 后的 latent
            injection_data: prepare_injection 返回的数据
            step_idx: 注入使用的 latent 索引（后置注入时为 denoising_step + 1）
            config: 注入配置
            
        Returns:
            注入后的 latent
        """
        if config is None:
            config = InjectionConfig()
        
        total_steps = injection_data["total_steps"]
        
        # timestep 维度：超过注入比例则跳过
        if not config.should_inject(step_idx, total_steps):
            return current_latent
        
        # 合并所有区域的 latent（取第一个区域的，简化处理）
        if not injection_data["latent_lists"]:
            return current_latent
        
        latent_list = injection_data["latent_lists"][0]
        # latent_list 有 N+1 项：[sigma_0, sigma_1, ..., sigma_{N-1}, clean(sigma=0)]
        # step_idx 范围: 0 ~ N
        idx = min(step_idx, len(latent_list) - 1)
        text_latent = latent_list[idx]
        
        mask = injection_data["mask_latent"]
        
        # 扩展 mask 到 latent 的 channel 维度
        mask = mask.expand_as(current_latent)
        
        # 空间混合
        s = config.mask_strength
        injected = current_latent * (1 - mask * s) + text_latent * mask * s
        
        # 可视化 injected latent
        if self.logger is not None:
            injected_img = self.decode_latent(injected)
            self.logger.save_image(
                injected_img,
                f"glyph_injected_step{step_idx}",
                caption=f"step={step_idx}/{total_steps}  mask_strength={s:.2f}",
                subfolder="glyph",
            )
        
        return injected


def create_glyph_injector(pipeline, device: str = "cuda", logger=None) -> GlyphInjector:
    """
    从 pipeline 创建 GlyphInjector
    
    Args:
        pipeline: ZImagePipeline 实例
        device: 设备
        logger: TTSLogger 实例（可选）
        
    Returns:
        GlyphInjector 实例
    """
    return GlyphInjector(
        vae=pipeline.vae,
        scheduler=pipeline.scheduler,
        device=device,
        dtype=pipeline.vae.dtype,
        logger=logger,
    )


if __name__ == "__main__":
    # 测试文字渲染
    injector = GlyphInjector.__new__(GlyphInjector)
    
    text = "x = (-b ± √(b²-4ac)) / 2a"
    img = GlyphInjector.render_text_template(injector, text, 512, 128)
    img.save("./test_text_render.png")
    print(f"文字渲染测试完成: /tmp/test_text_render.png")
    
    # 测试 mask 提取
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    mask = GlyphInjector.extract_text_mask(injector, img_bgr)
    cv2.imwrite("./test_text_mask.png", mask)
    print(f"Mask 提取测试完成: /tmp/test_text_mask.png")
