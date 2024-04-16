import cv2
import numpy as np

# 读取图像
image = cv2.imread('E:\\pythonProject\\opencv\\image\\f2.jpg')
print(image.shape)

# 将此图像二值化
ret, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# 应用高斯滤波进行模糊处理
blurred_image = cv2.GaussianBlur(binary_image, (101, 101), 3)

# 应用中值滤波进行平滑处理
blurred_image = cv2.medianBlur(binary_image, 25)

# 应用Canny边缘检测
edges = cv2.Canny(blurred_image, 50, 150)

# 创建结构元素（这里使用矩形，你也可以使用圆形或其他形状）
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

# 开闭运算(测试)
blurred_image = cv2.morphologyEx(blurred_image, cv2.MORPH_OPEN, kernel)

# 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 设置面积阈值
area_threshold_max = 10000  # 你可以根据实际情况调整这个值
area_threshold_min = 1000  # 你可以根据实际情况调整这个值

# 遍历所有轮廓
for contour in contours:
    # 计算轮廓的面积
    area = cv2.contourArea(contour)

    # 检查面积是否大于阈值
    if area > area_threshold_min and area < area_threshold_max:
        # 计算最小面积矩形
        rect = cv2.minAreaRect(contour)
        # 获取矩形的四个角点
        box = cv2.boxPoints(rect)
        box = np.intp(box)  # 将坐标转换为整数

        # 在原始图像上绘制矩形
        cv2.drawContours(image, [box], 0, (0, 0, 255), 3)

# 显示带有绘制轮廓的图像
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()