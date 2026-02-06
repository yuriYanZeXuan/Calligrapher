import cv2
import numpy as np

def straighten_text(image_path):
    """将倾斜的文字校正为水平正向"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    # 方法1：简单旋转（-4度）
    center = (w // 2, h // 2)
    angle = -10  # 逆时针旋转4度
    M_rotate = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M_rotate, (w, h), borderValue=0)
    cv2.imwrite("straighten_rotated.jpg", rotated)
    
    # 方法2：透视变换（更精确，考虑透视变形）
    # 原图四个角点（左低右高的四边形）
    pts_src = np.float32([
        [20, 25],   # 左上（实际偏下）
        [235, 15],  # 右上（实际偏上）
        [240, 110], # 右下
        [25, 120]   # 左下（实际偏下）
    ])
    
    # 目标矩形（水平正向）
    pts_dst = np.float32([
        [20, 20],   # 左上
        [235, 20],  # 右上
        [235, 115], # 右下
        [20, 115]   # 左下
    ])
    
    # 计算透视变换矩阵
    M_perspective = cv2.getPerspectiveTransform(pts_src, pts_dst)
    straightened = cv2.warpPerspective(img, M_perspective, (w, h), borderValue=0)
    cv2.imwrite("straighten_perspective.jpg", straightened)
    
    print("保存结果：")
    print("  - straighten_rotated.jpg: 简单旋转校正")
    print("  - straighten_perspective.jpg: 透视变换校正")
    
    return rotated, straightened

if __name__ == "__main__":
    image_path = "/Users/yanzexuan/code/Calligrapher/dev/patch_binary.jpg"
    straighten_text(image_path)
