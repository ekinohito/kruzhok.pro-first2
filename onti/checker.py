import os

from PIL import Image
import cv2
import pytesseract as tes


def test():
    image = cv2.imread('test.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.medianBlur(image, 3)
    # image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    temporary_file_name = '.temp.png'
    cv2.imwrite(temporary_file_name, image)

    text = tes.image_to_string(Image.open(temporary_file_name), lang='rus')
    # os.remove(temporary_file_name)
    print(text)


if __name__ == '__main__':
    test()
