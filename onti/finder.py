import cv2, imutils
import numpy as np
from skimage.filters import threshold_local

class DocNotFound(Exception):
    def __init__(self, text):
        self.txt = text


def scan(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    # show the original image and the edge detected image



    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.5 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    # show the contour (outline) of the piece of paper
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    image = imutils.resize(image, width=1000)
    cv2.imshow("Outline", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_leveled_document(image, template, debug = False): # На вход получает объект типа cv.imread()
    contours = scan(image)

    # for i in contours:
    #     if template_matched(i, image,template)
    #         return cropped(i,image)
    #
    # raise DocNotFound





if __name__ == '__main__':
    get_leveled_document(cv2.imread('2.jpg'),cv2.imread('1.jpg'),True)









