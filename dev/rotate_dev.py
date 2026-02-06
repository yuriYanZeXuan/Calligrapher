import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def get_available_font(size=100):
    """获取系统中可用的字体"""
    possible_fonts = [
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Geneva.ttf",
        "/System/Library/Fonts/NewYork.ttf",
        "/System/Library/Fonts/Monaco.ttf",
    ]
    for font_path in possible_fonts:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
    # 如果都失败，使用默认字体
    print("警告：使用默认字体")
    return ImageFont.load_default()

# ==========================================
# 主程序：处理 fail1.png 中的"九九乘法表"
# ==========================================
if __name__ == "__main__":
    # 1. 读取原图 (1024x1024)
    img_path = '/Users/yanzexuan/code/Calligrapher/dev/fail1.png'
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图片: {img_path}")
    
    H, W = img.shape[:2]
    print(f"原图尺寸: {W}x{H}")
    
    # 2. 根据相对坐标计算绝对坐标 [0.658, 0.16, 0.847, 0.489]
    # 格式: [x_min, y_min, x_max, y_max]
    x_min, y_min, x_max, y_max = 0.544, 0.062, 0.793, 0.183
    
    x1 = int(x_min * W)
    y1 = int(y_min * H)
    x2 = int(x_max * W)
    y2 = int(y_max * H)
    
    print(f"裁剪区域: ({x1}, {y1}) -> ({x2}, {y2})")
    
    # 3. 裁剪图像patch
    patch = img[y1:y2, x1:x2].copy()
    patch_h, patch_w = patch.shape[:2]
    print(f"Patch尺寸: {patch_w}x{patch_h}")
    
    # 4. 转换为灰度图
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    
    # 5. 大津法二值化
    # 先判断文字颜色：如果背景偏暗（黑板），文字是白色
    mean_val = np.mean(gray)
    print(f"灰度均值: {mean_val}")
    
    if mean_val < 127:
        # 背景偏暗（黑板），文字是白色
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 此时白色是文字，黑色是背景
        # 计算白色像素比例
        white_ratio = np.sum(binary == 255) / binary.size
        print(f"白色比例: {white_ratio:.3f}")
        if white_ratio < 0.5:
            # 白色较少，白色是文字
            text_mask = binary.copy()
        else:
            # 白色较多，需要反转
            text_mask = cv2.bitwise_not(binary)
    else:
        # 背景偏亮，文字是黑色
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        black_ratio = np.sum(binary == 255) / binary.size
        print(f"黑色(前景)比例: {black_ratio:.3f}")
        if black_ratio < 0.5:
            text_mask = binary.copy()
        else:
            text_mask = cv2.bitwise_not(binary)
    
    # 6. 形态学操作去除噪声
    kernel = np.ones((10, 10), np.uint8)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, kernel)
    
    # 7. 查找轮廓并获取凸包
    contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 合并所有轮廓点
    all_points = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:  # 过滤小噪声
            all_points.extend(cnt.reshape(-1, 2))
    
    if len(all_points) == 0:
        raise ValueError("未找到文字轮廓")
    
    all_points = np.array(all_points)
    
    # 8. 计算凸包
    hull = cv2.convexHull(all_points)
    
    # 9. 使用最小外接四边形（逼近凸包为四边形）
    # 方法：使用minAreaRect获取旋转矩形，然后获取box
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    
    print(f"检测到的四边形角点 (patch坐标):")
    for i, point in enumerate(box):
        print(f"  P{i}: ({point[0]}, {point[1]})")
    
    # 10. 将patch坐标转换回原图坐标
    box_in_original = box.copy()
    box_in_original[:, 0] += x1  # x坐标偏移
    box_in_original[:, 1] += y1  # y坐标偏移
    
    print(f"四边形角点 (原图坐标):")
    for i, point in enumerate(box_in_original):
        print(f"  P{i}: ({point[0]}, {point[1]})")
    
    # 11. 可视化结果
    result_img = img.copy()
    
    # 在原图上绘制四边形
    cv2.polylines(result_img, [box_in_original], isClosed=True, color=(0, 0, 255), thickness=3)
    
    # 绘制四个角点
    for i, point in enumerate(box_in_original):
        cv2.circle(result_img, tuple(point), 8, (0, 255, 0), -1)
        cv2.putText(result_img, str(i), (point[0]+10, point[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 12. 保存中间结果和最终结果
    cv2.imwrite("patch_original.jpg", patch)
    cv2.imwrite("patch_gray.jpg", gray)
    cv2.imwrite("patch_binary.jpg", binary)
    cv2.imwrite("patch_text_mask.jpg", text_mask)
    
    # 在patch上绘制检测结果
    patch_result = patch.copy()
    cv2.polylines(patch_result, [box], isClosed=True, color=(0, 0, 255), thickness=2)
    for i, point in enumerate(box):
        cv2.circle(patch_result, tuple(point), 5, (0, 255, 0), -1)
    cv2.imwrite("patch_result.jpg", patch_result)
    
    # 保存原图结果
    cv2.imwrite("result_with_quad.jpg", result_img)
    
    print(f"\n保存的文件:")
    print(f"  - patch_original.jpg: 裁剪的patch")
    print(f"  - patch_gray.jpg: 灰度图")
    print(f"  - patch_binary.jpg: 二值图")
    print(f"  - patch_text_mask.jpg: 文字掩码")
    print(f"  - patch_result.jpg: patch上的检测结果")
    print(f"  - result_with_quad.jpg: 原图上的四边形")
    
    # 13. 显示结果（可选）
    # 创建组合图
    fig_h = max(patch_h, 300)
    combined = np.zeros((fig_h, patch_w * 3, 3), dtype=np.uint8)
    
    # 调整所有图像到相同高度
    def resize_to_height(img, h):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        scale = h / img.shape[0]
        new_w = int(img.shape[1] * scale)
        return cv2.resize(img, (new_w, h))
    
    gray_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mask_color = cv2.cvtColor(text_mask, cv2.COLOR_GRAY2BGR)
    
    combined[:, :patch_w] = cv2.resize(patch, (patch_w, fig_h))
    combined[:, patch_w:patch_w*2] = cv2.resize(gray_color, (patch_w, fig_h))
    combined[:, patch_w*2:patch_w*3] = cv2.resize(mask_color, (patch_w, fig_h))
    
    cv2.imwrite("combined_process.jpg", combined)
    print(f"  - combined_process.jpg: 处理流程组合图")
