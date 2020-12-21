import cv2
import numpy as np

cv2.namedWindow("images")
image = cv2.imread("./gallery/railway3.jpg")
height = image.shape[0]
width = image.shape[1]


def colors(x):
    return


image = cv2.resize(image, (int(height*0.5), int(width*0.5)),
                   interpolation=cv2.INTER_NEAREST)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.createTrackbar("LH", "images", 0, 255, colors)
cv2.createTrackbar("LV", "images", 0, 255, colors)
cv2.createTrackbar("LS", "images", 248, 255, colors)
cv2.createTrackbar("UH", "images", 255, 255, colors)
cv2.createTrackbar("UV", "images", 255, 255, colors)
cv2.createTrackbar("US", "images", 255, 255, colors)

while True:
    region_of_interest_vertices = [
        (0, 0.9*height), (0.2*width, 0.35*height), (0.84*width, 0.35*height), (width, 0.7*height), (width, 0.9*height)]
    lh = cv2.getTrackbarPos("LH", "images")
    lv = cv2.getTrackbarPos("LV", "images")
    ls = cv2.getTrackbarPos("LS", "images")
    uh = cv2.getTrackbarPos("UH", "images")
    uv = cv2.getTrackbarPos("UV", "images")
    us = cv2.getTrackbarPos("US", "images")
    l_b = np.array([lh, lv, ls])
    u_b = np.array([uh, uv, us])
    mask = cv2.inRange(hsv, l_b, u_b)
    edges = cv2.Canny(mask, 255, 255, apertureSize=3)
    lines =
    # cv2.imshow("images", image)
    # cv2.imshow("hsv", hsv)
    cv2.imshow("mask", mask)
    cv2.imshow("edges", edges)

    k = cv2.waitKey(1) & 0XFF
    if k == ord("q"):
        break
cv2.destroyAllWindows()
