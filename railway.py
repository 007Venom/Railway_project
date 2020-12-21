import cv2
import numpy as np


def colors(x):
    return


cv2.namedWindow("images")
cv2.createTrackbar("LH", "images", 0, 255, colors)
cv2.createTrackbar("LV", "images", 0, 255, colors)
cv2.createTrackbar("LS", "images", 248, 255, colors)
cv2.createTrackbar("UH", "images", 255, 255, colors)
cv2.createTrackbar("UV", "images", 255, 255, colors)
cv2.createTrackbar("US", "images", 255, 255, colors)
cv2.createTrackbar("TH1", "images", 255, 255, colors)
cv2.createTrackbar("TH2", "images", 255, 255, colors)
cv2.createTrackbar("Blur", "images", 0, 7, colors)


image = cv2.imread("./gallery/railway3.jpg")

height = image.shape[0]
width = image.shape[1]
image = cv2.resize(image, (int(height*0.5), int(width*0.5)),
                   interpolation=cv2.INTER_NEAREST)
height = image.shape[0]
width = image.shape[1]
while True:
    region_of_interest_vertices = [
        (0, 0.9*height), (0.2*width, 0.35*height), (0.84*width, 0.35*height), (width, 0.7*height), (width, 0.9*height)]

    lh = cv2.getTrackbarPos("LH", "images")
    lv = cv2.getTrackbarPos("LV", "images")
    ls = cv2.getTrackbarPos("LS", "images")
    uh = cv2.getTrackbarPos("UH", "images")
    uv = cv2.getTrackbarPos("UV", "images")
    us = cv2.getTrackbarPos("US", "images")
    blur = cv2.getTrackbarPos("Blur", "images")
    th1 = cv2.getTrackbarPos("TH1", "images")
    th2 = cv2.getTrackbarPos("TH2", "images")
    l_b = np.array([lh, lv, ls])
    u_b = np.array([uh, uv, us])

    def draw_the_lines(img, lines):
        img = np.copy(img)
        blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1, y1), (x2, y2),
                         (0, 255, 0), thickness=8)
        img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
        return img

    def region_of_interest(msk, vertices):
        mask = np.zeros_like(msk)
        match_mask_color = 255
        cv2.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv2.bitwise_and(msk, mask)
        return masked_image
    cropped_image = region_of_interest(
        image, np.array([region_of_interest_vertices], np.int32))
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv, l_b, u_b)
    res = cv2.bitwise_and(image, image, mask=mask2)
    edges = cv2.Canny(mask2, th1, th2, apertureSize=3)
    gaussian_image = cv2.GaussianBlur(edges, (7, 7), 1, sigmaY=0)
    lines = cv2.HoughLinesP(edges, rho=6, theta=np.pi/60,
                            threshold=250, lines=np.array([]), minLineLength=10, maxLineGap=100)
    image_with_lines = draw_the_lines(image, lines)
    cv2.imshow("cropped_image", image_with_lines)
    cv2.imshow("images", cropped_image)
    cv2.imshow("edges", edges)
    cv2.imshow("mask", mask2)
    cv2.imshow("res", res)
    k = cv2.waitKey(1) & 0XFF
    if k == ord("q"):
        break
cv2.destroyAllWindows()
