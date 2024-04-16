import cv2
import numpy as np

# 读取图像
image = cv2.imread('E:\\pythonProject\\opencv\\image\\f3.jpg')
# print(image.shape)

# 转换至HSV颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义红色的HSV范围(一)
lower_red1 = np.array([120, 20, 50])
upper_red1 = np.array([255, 255, 255])

# 定义红色的HSV范围(二)
lower_red2 = (0, 70, 50)  # 低阈值
upper_red2 = (10, 255, 255)  # 高阈值


# 创建第一个掩码
mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)

# 创建第二个掩码
mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

# 合并两个掩码
combined_mask = cv2.bitwise_or(mask1, mask2)

# 形态学操作 - 腐蚀和膨胀
kernel = np.ones((5, 5), np.uint8)
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

# 寻找轮廓
contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# # 绘制轮廓
# image = cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

# #测试红色区域大小
# cv2.rectangle(image, (172,120), (298,230), (0,0,255), 5, 16)

# 遍历所有轮廓
for contour in contours:
    # 计算轮廓的最小矩形边界
    x, y, width, height = cv2.boundingRect(contour)

    # 计算轮廓面积
    area = width * height

    # 设置一个最小面积阈值来忽略小的噪声
    if area > 2000:  # 这个值可以根据实际情况调整
        # 绘制矩形边界
        cv2.rectangle(image, (x, y), (x + width, y + height), (100, 255, 0), 3)

# 显示带有圈出的红色物体的图像
cv2.imshow('Detected Red Objects', image)
cv2.waitKey(0)
cv2.destroyAllWindows()