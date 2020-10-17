import imutils, cv2
import numpy as np
from math import atan, pi, sqrt
from skimage.filters import threshold_local

class DocNotFound(Exception):
    def __init__(self, text):
        self.txt = text


def fourCornersSort(pts):#сортировка точек контура против часовой
    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)
    return np.array([pts[np.argmin(summ)],
                     pts[np.argmax(diff)],
                     pts[np.argmax(summ)],
                     pts[np.argmin(diff)]])


wid = 500
def scan(image): # принимает массив с cv2.imread()

    orig = image.copy()

    image = cv2.bilateralFilter(image, 9, 75, 75)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # H = 255 S = 80 V = 175
    H = 255
    S = 100
    V = 150

    white = cv2.inRange(hsv, (0, 0, V), (H, S, 255))

    kernel = np.ones((100, 100), np.uint8)#!!!!!!!!!!!!!!!!!!!!!
    white = cv2.dilate(white, kernel, iterations=1)

    dst = cv2.bitwise_and(image, image, mask=white)
    dst = cv2.copyMakeBorder(dst, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    adap = cv2.adaptiveThreshold(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 115, 4)

    adap = cv2.bilateralFilter(adap, 9, 75, 75)
    adap = cv2.medianBlur(adap, 11)
    edge = cv2.Canny(adap, 100, 200)


    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plos = 1000000
    apr = 0.03
    approx = 0
    for c, i in enumerate(contours):
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, apr * peri, True)
        if len(approx) == 4:
            if cv2.contourArea(approx) > plos:
                break

    approx = fourCornersSort(np.array([i[0] for i in approx]))

    height = max(np.linalg.norm(approx[0] - approx[1]),
                 np.linalg.norm(approx[2] - approx[3]))
    width = max(np.linalg.norm(approx[1] - approx[2]),
                np.linalg.norm(approx[3] - approx[0]))

    tPoints = np.array([[0, 0],
                        [0, height],
                        [width, height],
                        [width, 0]], np.float32)


    if approx.dtype != np.float32:
        approx = approx.astype(np.float32)

    M = cv2.getPerspectiveTransform(approx, tPoints)
    newImage = cv2.warpPerspective(orig, M, (int(width), int(height)))
    return newImage

