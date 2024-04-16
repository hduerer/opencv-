import cv2
import numpy as np

# 读取图像

image = cv2.imread('E:\\pythonProject\\opencv\\image\\f1.jpg', cv2.IMREAD_GRAYSCALE)

# print(image.shape)

# 应用高斯模糊，减少噪声和细节
image = cv2.GaussianBlur(image, (3, 3), 0)

# 使用Canny边缘检测器找到边缘
edges = cv2.Canny(image, 50, 150)

# 使用霍夫圆变换检测圆形
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                           param1=50, param2=30, minRadius=0, maxRadius=200)

# 确保检测到了圆形
if circles is not None:
    circles = np.uint16(np.around(circles))

    bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    print(bgr_image.shape)

    for i in circles[0, :]:
        # 画圆心
        # cv2.circle(image, (i[0], i[1]), 3, (0, 255, 0), 3)
        # 画圆轮廓
        cv2.circle(bgr_image, (i[0], i[1]), i[2], (0, 255, 255), -1)

# 显示结果图像
cv2.imshow('Detected Circles', bgr_image)
cv2.waitKey(0)
cv2.destroyAllWindows()