import cv2
import numpy as np


video = cv2.VideoCapture("./gallery/Pexels Videos 2119145.mp4")
cv2.namedWindow("frame")


def colors(x):
    return


cv2.createTrackbar("LH", "frame", 0, 255, colors)
cv2.createTrackbar("LV", "frame", 0, 255, colors)
cv2.createTrackbar("LS", "frame", 193, 255, colors)
cv2.createTrackbar("UH", "frame", 255, 255, colors)
cv2.createTrackbar("UV", "frame", 24, 255, colors)
cv2.createTrackbar("US", "frame", 255, 255, colors)
cv2.createTrackbar("TH1", "frame", 255, 255, colors)
cv2.createTrackbar("TH2", "frame", 255, 255, colors)
cv2.createTrackbar("TH3", "frame", 35, 255, colors)
frame_counter = 0
while (video.isOpened()):
    region_of_interest = np.array([[0.34*500, 500], [0.37*500, 0.6*500],
                                   [0.54*500, 0.6*500], [0.62*500, 500]])
    frame_counter = frame_counter+1
    if frame_counter == video.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_counter = 0
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = video.read()

    def regionOfInterest(msk, vertices):
        mask = np.zeros_like(msk)
        match_mask_color = 255
        cv2.fillPoly(mask, np.int32([vertices]), match_mask_color)
        masked_image = cv2.bitwise_and(msk, mask)
        return masked_image

    def draw_the_lines(img, lines):
        img = np.copy(img)
        blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1, y1), (x2, y2),
                         (0, 255, 0), thickness=8)
        img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
        return img

    frame = cv2.resize(frame, (500, 500))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lh = cv2.getTrackbarPos("LH", "frame")
    lv = cv2.getTrackbarPos("LV", "frame")
    ls = cv2.getTrackbarPos("LS", "frame")
    uh = cv2.getTrackbarPos("UH", "frame")
    uv = cv2.getTrackbarPos("UV", "frame")
    us = cv2.getTrackbarPos("US", "frame")
    th1 = cv2.getTrackbarPos("TH1", "frame")
    th2 = cv2.getTrackbarPos("TH2", "frame")
    th3 = cv2.getTrackbarPos("TH3", "frame")

    l_b = np.array([lh, lv, ls])
    u_b = np.array([uh, uv, us])

    mask = cv2.inRange(hsv, l_b, u_b)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cropped_image = regionOfInterest(mask, region_of_interest)
    edges = cv2.Canny(cropped_image, th1, th2, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=6, theta=np.pi/60, threshold=th3,
                            lines=np.array([]), minLineLength=10, maxLineGap=100)
    image_with_lines = draw_the_lines(frame, lines)
    cv2.imshow("image with lines", image_with_lines)
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("hsv", hsv)
    cv2.imshow("edges", edges)
    if cv2.waitKey(1) & 0XFF == ord("q"):
        break
video.release()
cv2.destroyAllWindows()
