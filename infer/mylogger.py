"""
TTS 日志管理模块
- 记录代码位置、时间、输出到文本日志
- 保存图片并以 prompt + 评分作为 caption
- 每次运行自动创建带时间戳的子目录
"""

import logging
import inspect
import json
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


LOG_ROOT = Path(__file__).parent.parent / "logs"


def _get_caller_info(stack_level: int = 2) -> str:
    """获取调用者的文件名和行号"""
    frame = inspect.stack()[stack_level]
    filename = Path(frame.filename).name
    return f"{filename}:{frame.lineno}"


class TTSLogger:
    """Test Time Scaling 日志记录器"""

    def __init__(self, run_name: str = None):
        # 运行目录: logs/<timestamp>_<run_name>/
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{ts}_{run_name}" if run_name else ts
        self.run_dir = LOG_ROOT / folder_name
        self.img_dir = self.run_dir / "images"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir.mkdir(parents=True, exist_ok=True)

        # 文本日志
        self._log_path = self.run_dir / "run.log"
        self._logger = logging.getLogger(f"tts.{ts}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        # 文件 handler
        fh = logging.FileHandler(self._log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
        self._logger.addHandler(fh)
        # 控制台 handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
        self._logger.addHandler(ch)

        # 结构化记录（JSON Lines）
        self._jsonl_path = self.run_dir / "records.jsonl"

        self.info(f"日志目录: {self.run_dir}")

    # ---------- 文本日志 ----------

    def info(self, msg: str):
        caller = _get_caller_info()
        self._logger.info(f"[{caller}] {msg}")

    def warn(self, msg: str):
        caller = _get_caller_info()
        self._logger.warning(f"[{caller}] {msg}")

    def error(self, msg: str):
        caller = _get_caller_info()
        self._logger.error(f"[{caller}] {msg}")

    # ---------- 图片保存 ----------

    def save_image(
        self,
        image: Image.Image,
        name: str,
        caption: str = "",
        subfolder: str = None,
    ) -> Path:
        """保存图片，可选添加 caption 水印

        Args:
            image: PIL Image
            name: 文件名（不含后缀）
            caption: 图片底部标注文字（prompt / 评分等）
            subfolder: 可选子文件夹
        Returns:
            保存路径
        """
        out_dir = self.img_dir / subfolder if subfolder else self.img_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"{name}.png"

        if caption:
            image = self._add_caption(image, caption)

        image.save(save_path, format="PNG")
        self.info(f"图片已保存: {save_path.relative_to(self.run_dir)}")
        return save_path

    # ---------- 结构化记录 ----------

    def log_vlm_score(
        self,
        stage: str,
        candidate_idx: int,
        prompt: str,
        score: float,
        image: Image.Image = None,
        extra: dict = None,
    ):
        """记录一次 VLM 评分事件，同时保存图片

        Args:
            stage: 搜索阶段名称，如 "early_stop", "final"
            candidate_idx: 候选编号
            prompt: 送给 VLM 的提示词
            score: VLM 评分
            image: 评分用的图像（会保存并标注 caption）
            extra: 其他信息
        """
        record = {
            "time": datetime.now().isoformat(),
            "stage": stage,
            "candidate_idx": candidate_idx,
            "prompt": prompt,
            "score": score,
        }
        if extra:
            record.update(extra)

        img_name = f"{stage}_cand{candidate_idx}"
        caption = f"[{stage}] cand={candidate_idx}  score={score:.2f}\n{prompt[:120]}"

        if image is not None:
            img_path = self.save_image(image, img_name, caption=caption, subfolder=stage)
            record["image_path"] = str(img_path.relative_to(self.run_dir))

        # 追加 JSONL
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.info(f"[{stage}] 候选{candidate_idx}  评分={score:.2f}")

    # ---------- 内部工具 ----------

    # 支持中文的字体搜索列表（Linux 服务器 → macOS 兜底）
    _FONT_CANDIDATES = [
        # Linux Noto CJK（最常见的服务器中文字体）
        "/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/Calligrapher/baselines/anytext/font/Arial_Unicode.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/OTF/NotoSansCJK-Regular.ttc",
        # Linux WenQuanYi
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc",
        # Linux SimHei / SimSun
        "/usr/share/fonts/truetype/SimHei.ttf",
        "/usr/share/fonts/chinese/SimHei.ttf",
        # macOS 中文字体
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        # 最后兜底的纯英文字体
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    _font_cache = None

    @classmethod
    def _load_font(cls, size: int = 18):
        """加载支持中文的字体，结果缓存"""
        if cls._font_cache is not None:
            return cls._font_cache.font_variant(size=size)

        for path in cls._FONT_CANDIDATES:
            if Path(path).exists():
                try:
                    cls._font_cache = ImageFont.truetype(path, size)
                    return cls._font_cache
                except OSError:
                    continue

        cls._font_cache = ImageFont.load_default()
        return cls._font_cache

    @staticmethod
    def _add_caption(image: Image.Image, text: str, bar_height: int = 40) -> Image.Image:
        """在图片底部添加白色条带 + caption 文字"""
        w, h = image.size
        n_lines = text.count("\n") + 1
        total_bar = bar_height * n_lines

        new_img = Image.new("RGB", (w, h + total_bar), "white")
        new_img.paste(image, (0, 0))

        draw = ImageDraw.Draw(new_img)
        font = TTSLogger._load_font(size=18)
        draw.text((10, h + 8), text, fill="black", font=font)
        return new_img
