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

from .formula_helper import (
    is_latex,
    plaintext_to_latex,
    render_latex,
    render_plaintext,
    render_formula,
    get_available_font,
    calculate_font_size,
)


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

    基础注入:
        mask_strength: 空间混合强度 (0-1)，控制文字 latent 在 mask 区域的混合权重
        timestep_ratio: 时间步注入比例 (0-1)，仅在前 X% 的去噪步骤中注入
        inject_last_only: 仅在最后一步注入（忽略 timestep_ratio，直接在去噪完成后粘贴字形）

    方案 A - 频率分解注入:
        freq_decompose: 启用后只注入高频（笔画结构），保留模型生成的低频（颜色/风格）
        freq_kernel_size: 高斯模糊核大小，用于分离高低频

    方案 C - 递减注入强度:
        strength_schedule: 注入强度随时间步的衰减策略 ("constant"/"linear"/"cosine")

    方案 D - 反向注意力抑制:
        attn_suppress_scale: 非 glyph 区域对 text token 的注意力抑制倍率 (<1 抑制, 1 不抑制)

    Attention Enhancement:
        attn_enhance_scale: attention reweighting 倍率
        attn_enhance_timestep_ratio: 仅在前 X% 的去噪时间步中激活增强
        attn_enhance_layers: 激活增强的 transformer layer 索引列表，None = 所有层
        attn_enhance_text_to_image: 增强 text→image 方向
        attn_enhance_image_to_text: 增强 image→text 方向
    """
    mask_strength: float = 1.
    timestep_ratio: float = 1.
    inject_last_only: bool = True  # 仅在最后一步注入（去噪完成后粘贴字形）

    # 方案 A: 频率分解注入
    freq_decompose: bool = False
    freq_kernel_size: int = 5

    # 方案 C: 递减注入强度调度
    strength_schedule: str = "constant"  # "constant" / "linear" / "cosine"

    # 方案 D: 反向注意力抑制
    attn_suppress_scale: float = 1.0  # <1 抑制非 glyph 区域，如 0.1

    # Prompt-Latent Attention Enhancement
    attn_enhance_scale: float = 1.
    attn_enhance_timestep_ratio: float = 1.
    attn_enhance_layers: Optional[list[int]] = None
    attn_enhance_text_to_image: bool = True
    attn_enhance_image_to_text: bool = True

    def should_inject(self, step_idx: int, total_steps: int) -> bool:
        """判断当前步是否需要注入。

        step_idx 从 1 开始（后置注入：denoising_step + 1），total_steps = 总去噪步数。
        inject_last_only=True 时仅在最后一步（step_idx == total_steps）注入。
        """
        if self.inject_last_only:
            return step_idx >= total_steps
        if self.timestep_ratio >= 1.0:
            return True
        return step_idx < int(total_steps * self.timestep_ratio)
    
    def get_strength(self, step_idx: int, total_steps: int) -> float:
        """获取当前步的注入强度（方案 C）"""
        base = self.mask_strength
        if self.strength_schedule == "constant":
            return base
        t = step_idx / max(total_steps - 1, 1)  # 0→1
        if self.strength_schedule == "linear":
            return base * (1.0 - t)
        if self.strength_schedule == "cosine":
            import math
            return base * 0.5 * (1.0 + math.cos(math.pi * t))
        return base
    
    @property
    def attn_enhance_enabled(self) -> bool:
        """注意力增强是否启用"""
        return self.attn_enhance_text_to_image or self.attn_enhance_image_to_text


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
        background_color: str = "black",
        text_color: str = "white",
        force_latex: bool = False,
        font_weight: str = "regular",
        font_path: Optional[str] = None,
    ) -> Image.Image:
        """渲染文字模板图像，支持纯文本和 LaTeX 公式。

        自动检测 LaTeX 内容（\\frac, \\int, ^{}, 等），使用 matplotlib
        渲染复杂数学公式。纯文本使用 PIL 字体渲染。

        Args:
            text: 渲染文本内容
            width, height: 输出图像尺寸
            background_color: 背景颜色
            text_color: 文字颜色
            force_latex: 强制 LaTeX 模式
            font_weight: 字体粗细 ("light"/"regular"/"bold")
            font_path: 自定义字体路径（可选）
        """
        return render_formula(
            text, width, height, text_color, background_color,
            force_latex, font_weight=font_weight, font_path=font_path,
        )
    
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
            latent = latent.to(self.vae.dtype)
            latent = (latent / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latent, return_dict=False)[0]
            
        # 转换为 PIL Image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().float().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
        
        return Image.fromarray(image)
    
    @staticmethod
    def _freq_decompose_inject(
        current: torch.Tensor,
        template: torch.Tensor,
        mask: torch.Tensor,
        strength: float,
        kernel_size: int = 5,
        return_debug: bool = False,
    ):
        """方案 A: 频率分解注入。

        只注入 template 的高频分量（笔画边缘），保留 current 的低频分量（颜色/风格）。

        Args:
            return_debug: 为 True 时返回 (injected, debug_dict)，debug_dict 含 template_lf/hf, current_lf/hf, blended_hf 用于可视化。
        """
        import torch.nn.functional as F

        pad = kernel_size // 2

        def blur(x):
            return F.avg_pool2d(
                F.pad(x, [pad] * 4, mode="reflect"),
                kernel_size, stride=1,
            )

        template_lf = blur(template)
        template_hf = template - template_lf

        current_lf = blur(current)
        current_hf = current - current_lf

        blended_hf = current_hf * (1 - mask * strength) + template_hf * (mask * strength)
        injected = current_lf + blended_hf

        if return_debug:
            debug = {
                "template_lf": template_lf,
                "template_hf": template_hf,
                "current_lf": current_lf,
                "current_hf": current_hf,
                "blended_hf": blended_hf,
            }
            return injected, debug
        return injected
    
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
            
            # 安全保障：确保渲染结果尺寸精确匹配 region
            if text_img.size != (region_width, region_height):
                text_img = text_img.resize((region_width, region_height), Image.LANCZOS)

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
        # INTER_AREA 对每个 latent 像素取源像素块均值，笔画不会因采样点偏移而丢失
        # 再用阈值二值化：块内有 >1% 的笔画像素即视为需要注入
        latent_h = 2 * (height // (self.vae_scale_factor * 2))
        latent_w = 2 * (width // (self.vae_scale_factor * 2))
        mask_area = cv2.resize(full_mask, (latent_w, latent_h), interpolation=cv2.INTER_AREA)
        mask_latent = (mask_area > 2).astype(np.float32)  # 阈值 ~1% of 255
        mask_latent = torch.from_numpy(mask_latent).unsqueeze(0).unsqueeze(0).to(self.device)
        
        injection_data["mask_latent"] = mask_latent
        injection_data["full_mask"] = full_mask
        injection_data["total_steps"] = len(timesteps)
        
        return injection_data
    
    def prepare_injection_from_plan(
        self,
        typography_plan: dict,
        image_size: Tuple[int, int],
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> dict:
        """根据 VLM 排版规划渲染字形模版并准备注入数据。

        遍历 plan["text_regions"] 中每个 region（由 VLM 自主规划），
        读取其 bbox/color/font_weight/font_size_ratio/is_latex 等排版信息，
        渲染字形模版后执行 mask 提取 + encode + inversion 流程。

        Args:
            typography_plan: VLM 返回的排版规划 JSON dict
            image_size: 图像尺寸 (width, height)
            noise: 初始噪声
            timesteps: 时间步列表

        Returns:
            与 prepare_injection() 相同格式的注入数据 dict
        """
        width, height = image_size
        full_mask = np.zeros((height, width), dtype=np.uint8)

        # 从 image_analysis 提取默认背景色，作为合成模版的底色
        analysis = typography_plan.get("image_analysis", {})
        dominant = analysis.get("dominant_colors", ["#000000"])
        default_bg = dominant[0] if dominant else "#000000"

        # 合成全图模版：所有 region 的文字画在同一张图上，编码一次 latent
        combined_template = Image.new("RGB", (width, height), default_bg)

        injection_data = {
            "latent_lists": [],
            "regions": [],
        }

        for idx, region_spec in enumerate(typography_plan.get("text_regions", [])):
            content = region_spec["content"]
            bbox = region_spec["bbox"]  # 归一化坐标 [x_min, y_min, x_max, y_max]

            # 转换为像素坐标
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)
            region_width = max(x2 - x1, 1)
            region_height = max(y2 - y1, 1)

            # 读取排版参数
            text_color = region_spec.get("color", "#FFFFFF")
            bg_color = region_spec.get("background_color", default_bg)
            font_weight = region_spec.get("font_weight", "regular")
            force_latex = region_spec.get("is_latex", False)

            # 渲染字形模版
            text_img = self.render_text_template(
                content,
                region_width,
                region_height,
                background_color=bg_color,
                text_color=text_color,
                force_latex=force_latex,
                font_weight=font_weight,
            )

            # 安全保障：确保渲染结果尺寸精确匹配 region
            if text_img.size != (region_width, region_height):
                text_img = text_img.resize((region_width, region_height), Image.LANCZOS)

            # 提取 mask
            text_array = np.array(text_img)
            text_mask = self.extract_text_mask(text_array)

            # 将 mask 放入完整图像
            full_mask[y1:y2, x1:x2] = text_mask

            # 将该 region 的渲染结果粘贴到合成模版上
            combined_template.paste(text_img, (x1, y1))

            injection_data["regions"].append((x1, y1, x2, y2))

            # 日志：保存每个 region 的单独渲染
            if self.logger is not None:
                caption = (
                    f"[plan] text=\"{content}\"  bbox=({x1},{y1},{x2},{y2})  "
                    f"size={region_width}x{region_height}\n"
                    f"weight={font_weight}  color={text_color}  bg={bg_color}  "
                    f"latex={force_latex}"
                )
                self.logger.save_image(
                    text_img, f"glyph_plan_region_{idx}_text",
                    caption=caption, subfolder="glyph",
                )
                mask_pil = Image.fromarray(text_mask)
                self.logger.save_image(
                    mask_pil.convert("RGB"), f"glyph_plan_region_{idx}_mask",
                    caption=caption, subfolder="glyph",
                )

        # 编码合成模版为 latent（所有 region 在一张图上），只编码/inversion 一次
        combined_latent = self.encode_image(combined_template)
        combined_latent_list = self.compute_inversion_latents(combined_latent, noise, timesteps)
        injection_data["latent_lists"] = [combined_latent_list]

        # 日志：保存合成模版全图
        if self.logger is not None:
            n_regions = len(injection_data["regions"])
            self.logger.save_image(
                combined_template, "glyph_plan_combined_template",
                caption=f"combined template: {n_regions} regions  size={width}x{height}",
                subfolder="glyph",
            )

        # mask 下采样到 latent 空间
        latent_h = 2 * (height // (self.vae_scale_factor * 2))
        latent_w = 2 * (width // (self.vae_scale_factor * 2))
        mask_area = cv2.resize(full_mask, (latent_w, latent_h), interpolation=cv2.INTER_AREA)
        mask_latent = (mask_area > 2).astype(np.float32)
        mask_latent = torch.from_numpy(mask_latent).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 保存完整分辨率 mask 到 logs
        if self.logger is not None:
            full_mask_pil = Image.fromarray(full_mask)
            coverage = (full_mask > 0).sum() / full_mask.size
            self.logger.save_image(
                full_mask_pil.convert("RGB"),
                "full_resolution_mask",
                caption=f"Full resolution mask | Shape: {full_mask.shape} | Coverage: {coverage:.2%}",
                subfolder="glyph",
            )

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
        
        # 使用 latent 空间的二值 mask
        mask = injection_data["mask_latent"]
        
        # 扩展 mask 到 latent 的 channel 维度
        mask = mask.expand_as(current_latent)
        if self.logger is not None:
            mask_np = (mask[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_np)
            self.logger.save_image(
                mask_pil.convert("RGB"),
                f"glyph_latent_mask",
                caption=f"latent mask  shape={list(mask.shape)}  coverage={mask.float().mean():.4f}",
                subfolder="glyph",
            )
        # 获取当前步的注入强度（方案 C: 递减调度）
        s = config.get_strength(step_idx, total_steps)
        
        # 方案 A: 频率分解 — 只注入高频（笔画结构），保留模型的低频（风格/颜色）
        if config.freq_decompose:
            need_freq_vis = (
                self.logger is not None and step_idx == 0
            )  # 仅第一步保存频率分解可视化
            if need_freq_vis:
                injected, freq_debug = self._freq_decompose_inject(
                    current_latent, text_latent, mask, s, config.freq_kernel_size,
                    return_debug=True,
                )
                for name, lat in freq_debug.items():
                    img = self.decode_latent(lat)
                    self.logger.save_image(
                        img,
                        f"freq_decompose_{name}",
                        caption=f"kernel={config.freq_kernel_size} step={step_idx} strength={s:.3f}",
                        subfolder="glyph",
                    )
            else:
                injected = self._freq_decompose_inject(
                    current_latent, text_latent, mask, s, config.freq_kernel_size,
                )
        else:
            # 原始全频注入
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
    # 公式渲染测试请使用: python -m infer.formula_helper
    print("请运行 python -m infer.formula_helper 进行公式渲染测试")
