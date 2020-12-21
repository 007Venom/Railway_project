import cv2
import numpy as np


image = cv2.imread("./gallery/railway3.jpg")
image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_NEAREST)


def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2),
                     (0, 255, 0), thickness=8)
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img


l_b = np.array([0, 0, 239])
u_b = np.array([255, 43, 255])
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, l_b, u_b)

res = cv2.bitwise_and(image, image, mask=mask)
#gaussian_image = cv2.GaussianBlur(res, (5, 5), sigmaX=1, sigmaY=0)
edges = cv2.Canny(res, 0, 255, apertureSize=3)

lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                        100, minLineLength=100, maxLineGap=10)
image_with_lines = draw_the_lines(image, lines)


cv2.imshow("images", image_with_lines)
cv2.imshow("edges", edges)
cv2.imshow("mask", mask)
cv2.imshow("res", res)
cv2.waitKey(100000)
cv2.destroyAllWindows()
