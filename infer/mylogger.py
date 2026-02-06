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

    @staticmethod
    def _add_caption(image: Image.Image, text: str, bar_height: int = 60) -> Image.Image:
        """在图片底部添加白色条带 + caption 文字"""
        w, h = image.size
        # 多行文本时增加高度
        n_lines = text.count("\n") + 1
        total_bar = bar_height * n_lines

        new_img = Image.new("RGB", (w, h + total_bar), "white")
        new_img.paste(image, (0, 0))

        draw = ImageDraw.Draw(new_img)
        # 尝试加载等宽字体，失败则用默认
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
        except OSError:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 16)
            except OSError:
                font = ImageFont.load_default()

        draw.text((10, h + 8), text, fill="black", font=font)
        return new_img
