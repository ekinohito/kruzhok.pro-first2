import imutils, cv2
import numpy as np
from math import atan, pi, sqrt
from skimage.filters import threshold_local

class DocNotFound(Exception):
    def __init__(self, text):
        self.txt = text


def fourCornersSort(pts):#сортировка точек контура против часовой
    print(pts)
    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)
    return np.array([pts[np.argmin(summ)],
                     pts[np.argmax(diff)],
                     pts[np.argmax(summ)],
                     pts[np.argmin(diff)]])


wid = 500
def scan(image):
    image = cv2.imread('2.jpg')
    orig = image.copy()
    image = cv2.bilateralFilter(image, 9, 75, 75)


    cv2.namedWindow('1')
    # cv2.createTrackbar('s', '1', 0, 255, nothing)
    # cv2.createTrackbar('h', '1', 0, 255, nothing)
    # cv2.createTrackbar('v', '1', 0, 255, nothing)

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


    # black = cv2.morphologyEx(black, cv2.MORPH_OPEN, kernel)
    # black = cv2.morphologyEx(black, cv2.MORPH_GRADIENT, kernel)

    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plos = 1000000
    apr = 0.03
    for c, i in enumerate(contours):
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, apr * peri, True)
        if len(approx) == 4:
            if cv2.contourArea(approx) > plos:
                print(cv2.contourArea(approx))
                print(1)
                break
    cv2.drawContours(image, np.array(approx), -1, (0, 255, 0), 3)
    print(approx)
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

    # Wraping perspective
    M = cv2.getPerspectiveTransform(approx, tPoints)
    newImage = cv2.warpPerspective(orig, M, (int(width), int(height)))



    image = imutils.resize(image, width=wid)
    cv2.imshow('2', image)
    white = imutils.resize(white, width=wid)
    cv2.imshow('1', white)
    dst = imutils.resize(dst, width=wid)
    cv2.imshow('4', dst)
    newImage = imutils.resize(newImage, width=wid)
    cv2.imshow('3', newImage)
    edge = imutils.resize(edge, width=wid)
    cv2.imshow('5', edge)
    adap = imutils.resize(adap, width=wid)
    cv2.imshow('6', adap)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # while True:
    #
    #     h = cv2.getTrackbarPos('h', '1')
    #
    #     s = cv2.getTrackbarPos('s', '1')
    #
    #     v = cv2.getTrackbarPos('v', '1')
    #
    #
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     white = cv2.inRange(hsv, (0, 0, v), (h, s, 255))
    #     white = imutils.resize(white, width=wid)
    #     # image = imutils.resize(image, width=wid)
    #     cv2.imshow('1',white)
    #     k = cv2.waitKey(1) & 0xFF
    #     if k == 27:
    #         break

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # edged = cv2.Canny(gray, 75, 200)
    # # show the original image and the edge detected image
    #
    #
    #
    # cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    # # loop over the contours
    # for c in cnts:
    #     # approximate the contour
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.5 * peri, True)
    #     # if our approximated contour has four points, then we
    #     # can assume that we have found our screen
    #     if len(approx) == 4:
    #         screenCnt = approx
    #         break
    # # show the contour (outline) of the piece of paper
    # cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    # image = imutils.resize(image, width=1000)
    # cv2.imshow("Outline", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def get_leveled_document(image, template, debug = False): # На вход получает объект типа cv.imread()
    contours = scan(image)

    # for i in contours:
    #     if template_matched(i, image,template)
    #         return cropped(i,image)
    #
    # raise DocNotFound





if __name__ == '__main__':
    get_leveled_document(cv2.imread('2.jpg'),cv2.imread('1.jpg'),True)




"""

row, col = im.shape[:2]

black = cv2.inRange(hsv, (0, 0, 0), (180, 30, 160))
# cv2.imshow('23',black)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

edges = cv2.Canny(black, 0, 0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# black = cv2.morphologyEx(black, cv2.MORPH_OPEN, kernel)
# black = cv2.morphologyEx(black, cv2.MORPH_GRADIENT, kernel)

_, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
squares = []
plos = 30
apr = 0.03
for c, i in enumerate(contours):
    peri = cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i, apr * peri, True)
    if len(approx) == 4:
        if cv2.contourArea(approx) > plos:
            squares.append(approx)

cheked_sq = []
for i in squares:
    l = list(i)
    l.sort(key=lambda s: s[0][0] ** 2 + s[0][1] ** 2)
    r, c = im.shape[:2]
    mask = np.zeros((r, c, 1), np.uint8)
    cv2.fillPoly(mask, [i], (255, 255, 255))
    dst = cv2.bitwise_and(im, im, mask=mask)
    hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)

    white = cv2.inRange(hsv, (0, 0, 180), (180, 15, 255))

    edges = cv2.Canny(white, 0, 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sq1 = []

    for c, i in enumerate(contours):
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, apr * peri, True)
        if len(approx) == 4:
            if cv2.contourArea(approx) > plos:
                sq1.append(approx)

    for j in sq1:
        l = list(j)
        ll = l.copy()
        l.sort(key=lambda s: s[0][0] ** 2 + s[0][1] ** 2)
        pts1 = np.float32(l)
        pts2 = np.float32([[0, 0], [0, 100], [100, 0], [100, 100]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst1 = cv2.warpPerspective(im, M, (100, 100))
        hsv = cv2.cvtColor(dst1, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(dst1, cv2.COLOR_BGR2GRAY)
        white = cv2.inRange(hsv, (0, 0, 180), (180, 15, 255))

        edges = cv2.Canny(white, 0, 0)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sq2 = []
        for c, i in enumerate(contours):
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, apr * peri, True)
            if len(approx) == 4:
                if cv2.contourArea(approx) > plos:
                    sq2.append(approx)
        if len(sq2) == 1:
            cheked_sq.append([ll, 0])
        elif len(contours) != 0:
            if (cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 3, 1) is not None):
                cheked_sq.append([ll, 1])
center_coords = []
for i in cheked_sq:
    center_coords.append([int((i[0][0][0][0] + i[0][2][0][0]) / 2), int((i[0][0][0][1] + i[0][2][0][1]) / 2)])
center_x = int((center_coords[0][0] + center_coords[1][0] + center_coords[2][0]) // 4)
center_y = int((center_coords[0][1] + center_coords[1][1] + center_coords[2][1]) // 4)


def polar(m):
    x = m[0][0] - center_x
    y = m[0][1] - center_y
    if (x) != 0:
        if (y >= 0) and (x > 0):
            return (atan((y) / (x)))
        elif (y >= 0) and ((x) < 0):
            return (pi + atan((y) / (x)))
        elif (y <= 0) and ((x) < 0):
            return (pi + atan((y) / (x)))
        else:
            return (2 * pi + atan((y) / (x)))
    elif (y) > 0:
        return 1.57079633
    else:
        return 4.71238898


for i in range(len(cheked_sq)):
    cheked_sq[i][0] = center_coords[i].copy()
cheked_sq.sort(key=polar, reverse=True)
while cheked_sq[0][1] != 1:
    cheked_sq.append(cheked_sq.pop(0))

x = int(cheked_sq[0][0][0] + cheked_sq[2][0][0] - cheked_sq[1][0][0])
y = int(cheked_sq[0][0][1] + cheked_sq[2][0][1] - cheked_sq[1][0][1])

center_coords.append([x, y])

cheked_sq[1][1] = 2
cheked_sq[2][1] = 3
cheked_sq.append([[x, y], 4])

pts1 = np.float32([(int(i[0][0]), int(i[0][1])) for i in cheked_sq])
pts2 = np.float32([[250, 250], [500, 250], [500, 500], [250, 500]])

M = cv2.getPerspectiveTransform(pts1, pts2)

dst = cv2.warpPerspective(im, M, (750, 750))

qw = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
m1 = cv2.inRange(qw, (0, 80, 100), (10, 255, 255))
m2 = cv2.inRange(qw, (160, 80, 100), (179, 255, 255))
m3 = m1 + m2
M = cv2.moments(m3)

cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

print(cX - 245, cY - 245)

# cX = round(M["m10"] / M["m00"])
# cY = round(M["m01"] / M["m00"])
# a = sqrt(((cheked_sq[0][0][0]-cheked_sq[1][0][0])**2) +(cheked_sq[0][0][1]-cheked_sq[1][0][1])**2)
# b = sqrt(((cheked_sq[0][0][0]-cheked_sq[3][0][0])**2) +(cheked_sq[0][0][1]-cheked_sq[3][0][1])**2)
# c = sqrt(((cheked_sq[1][0][0]-cheked_sq[3][0][0])**2) +(cheked_sq[1][0][1]-cheked_sq[3][0][1])**2)
# d = sqrt(((cheked_sq[1][0][0]-cheked_sq[2][0][0])**2) +(cheked_sq[1][0][1]-cheked_sq[2][0][1])**2)
# e = sqrt(((cheked_sq[0][0][0]-cheked_sq[2][0][0])**2) +(cheked_sq[0][0][1]-cheked_sq[2][0][1])**2)
# cosa = (a**2+b**2-c**2)/(2*a*b)
# cosb = (a**2+d**2-e**2)/(2*a*d)
# print(cosb)
# print(acos(cosb)*180/pi)
# print(cosa)
# print(acos(cosa)*180/pi)



"""




