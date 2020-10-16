import os
import re
from typing import Tuple, List
from PIL import Image
import cv2
import pytesseract as tes


def prepare(s: str):
    return re.sub(r'[^а-я0-9]', '', s.lower())


class RequiredBlock:
    def __init__(self, pattern: str, threshold: int = 0):
        self.pattern = pattern
        self.threshold = threshold

    def score(self, text: str) -> Tuple[str, int]:
        match = re.search(self.pattern, text)
        if match:
            return text[match.regs[0][1]:], 0
        return '', 100

    def compare(self, text: str) -> Tuple[str, bool]:
        remaining, score = self.score(text)
        return remaining, score <= self.threshold


class RequiredText:
    def __init__(self, blocks: List[RequiredBlock]):
        self.blocks = blocks.copy()

    def score(self, text: str):
        result = 0
        for block in self.blocks:
            text, block_score = block.score(text)
            result += block_score
        return result

    def compare(self, text: str):
        for block in self.blocks:
            text, block_passed = block.score(text)
            if not block_passed:
                return False
        return True


def test():
    RequiredBlock('Согласие')
    image = cv2.imread('test.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.medianBlur(image, 3)
    # image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    temporary_file_name = '.temp.png'
    cv2.imwrite(temporary_file_name, image)

    text = tes.image_to_string(Image.open(temporary_file_name), lang='rus')
    # os.remove(temporary_file_name)
    prepared_text = prepare(text)
    print(prepared_text)




if __name__ == '__main__':
    test()
