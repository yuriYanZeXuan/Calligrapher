import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图片
img_path = '/Users/yanzexuan/code/Calligrapher/dev/fail1.png'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# "九九乘法表"五个字标题的四边形凸包顶点坐标
# [0.544, 0.062, 0.793, 0.183]
# 精确贴合每个字的实际边界
# points = np.array([
#     [542, 95],   # 左上 - 第一个"九"字左上角（降低y值贴合文字顶部）
#     [795, 65],    # 右上 - "表"字右上角
#     [795, 138],   # 右下 - "表"字右下角
#     [542, 180]    # 左下 - 第一个"九"字左下角
# ], dtype=np.int32)
# 根据比例坐标 [0.544, 0.062, 0.793, 0.183] 和 1024x1024画幅，生成四个顶点
ratio_bbox = [0.544, 0.062, 0.793, 0.183]
img_w, img_h = 1024, 1024

x1 = int(ratio_bbox[0] * img_w)
y1 = int(ratio_bbox[1] * img_h)
x2 = int(ratio_bbox[2] * img_w)
y2 = int(ratio_bbox[3] * img_h)

# 按常规顺序：左上，右上，右下，左下（假设矩形无旋转，坐标已归一化且已按此顺序）
points = np.array([
    [x1, y1],      # 左上
    [x2, y1],      # 右上
    [x2, y2],      # 右下
    [x1, y2]       # 左下
], dtype=np.int32)

# [ [542, 95], [795, 65], [795, 138], [542, 180] ]
# 创建图片副本用于绘制
img_with_quad = img.copy()

# 绘制四边形边框（红色，线宽3）
cv2.polylines(img_with_quad, [points], isClosed=True, color=(255, 0, 0), thickness=3)

# 绘制四个顶点（绿色圆点，半径5）
for i, point in enumerate(points):
    cv2.circle(img_with_quad, tuple(point), 5, (0, 255, 0), -1)
    # 标注点序号
    cv2.putText(img_with_quad, str(i+1), (point[0]+10, point[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 显示结果
plt.figure(figsize=(12, 12))
plt.imshow(img_with_quad)
plt.title('Title Bounding Box (Final)', fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.savefig('/Users/yanzexuan/code/Calligrapher/dev/fail1_title_bbox.png', dpi=150, bbox_inches='tight')
plt.close()

print("四边形顶点坐标（精确贴合'九九乘法表'）：")
print(f"P1 (左上): ({points[0][0]}, {points[0][1]})")
print(f"P2 (右上): ({points[1][0]}, {points[1][1]})")
print(f"P3 (右下): ({points[2][0]}, {points[2][1]})")
print(f"P4 (左下): ({points[3][0]}, {points[3][1]})")
print(f"\n保存结果至: /Users/yanzexuan/code/Calligrapher/dev/fail1_title_bbox.png")
