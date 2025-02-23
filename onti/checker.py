import json
import re
import sys
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Union

import cv2
import pytesseract as tes
from PIL import Image

from onti import finder


def prepare(s: str):
    """
    produces lowercase russian-only symbols
    :param s: given string
    :return: prepared string
    """
    return re.sub(r'[^а-я0-9]', '', s.lower())


class Requirement(ABC):
    """
    This abstract class corresponds to requirement. Any acceptable agreement should pass it's test.
    """

    @abstractmethod
    def score(self, text: str, data: Dict[str, Union[int, str]], image: Image) -> Tuple[str, int]:
        """
        scores text in some way
        :param image: whole image
        :param text: given text
        :param data: text data
        :return: remaining text and score
        """
        pass

    @abstractmethod
    def compare(self, text: str, data: Dict[str, Union[int, str]], image: Image) -> Tuple[str, bool]:
        """
        determines whether text contains requirement or not
        :param image: whole image
        :param text: given text
        :param data: text data
        :return: True if text is acceptable, False otherwise
        """
        pass


class RequiredText(Requirement):
    """
    This class corresponds to text data which should be represented in agreement.
    """

    def __init__(self, record: Dict[str, Union[int, str]]):
        """
        required text
        :param record: json record
        """
        self.pattern: str = record["str"]
        self.threshold: int = record.get("threshold", 0)

    def score(self, text: str, data: Dict[str, Union[int, str]], image: Image) -> Tuple[str, int]:
        match = re.search(self.pattern, text)
        if match:
            return text[match.regs[0][1]:], 0
        return '', 100

    def compare(self, text: str, data: Dict[str, Union[int, str]], image: Image) -> Tuple[str, bool]:
        remaining, score = self.score(text, data, image)
        return remaining, score <= self.threshold

    def __repr__(self):
        return f'text: {self.pattern}'


class RequiredForm(Requirement):
    """
    This class corresponds to some form in agreement which must be filled.
    """

    def __init__(self, record: Dict[str, Union[int, str]]):
        """
        required form
        :param record: json record
        """
        self.left_anchor: str = record['left']
        self.right_anchor: str = record['right']
        self.threshold: int = record['threshold']

    def score(self, text: str, data: Dict[str, Union[int, str]], image: Image) -> Tuple[str, int]:
        left_index, right_index = None, None
        for i, anchor_name in enumerate(data['text']):
            if re.search(self.left_anchor, anchor_name, re.IGNORECASE):
                left_index = i
                break
        if not left_index:
            return text, 0
        for i, anchor_name in enumerate(data['text'][left_index + 1:]):
            if re.search(self.right_anchor, anchor_name, re.IGNORECASE):
                right_index = left_index + 1 + i
                break
        if not right_index:
            return text, 0
        field = image.crop((data['left'][left_index] + data['width'][left_index], data['top'][left_index],
                            data['left'][right_index], data['top'][right_index] + data['height'][right_index]))
        # print(field.entropy())
        return text, field.entropy()

    def compare(self, text: str, data: Dict[str, Union[int, str]], image: Image) -> Tuple[str, bool]:
        remaining, score = self.score(text, data, image)
        return remaining, score * 100 >= self.threshold

    def __repr__(self):
        return f'form: "{self.left_anchor}" to "{self.right_anchor}"'


class RequiredOption(Requirement):
    def __init__(self, record):
        self.requirements = [process_requirement(requirement) for requirement in record["requirements"]]

    def score(self, text: str, data: Dict[str, Union[int, str]], image: Image) -> Tuple[str, int]:
        result = 0
        for requirement in self.requirements:
            text, requirement_score = requirement.score(text, data, image)
            result += requirement_score
        return text, result

    def compare(self, text: str, data: Dict[str, Union[int, str]], image: Image) -> Tuple[str, bool]:
        for requirement in self.requirements:
            text, requirement_passed = requirement.compare(text, data, image)
            # print(f'block({requirement}) passed: {requirement_passed}')
            if requirement_passed:
                return text, True
        return text, False

    def __repr__(self):
        return f"option: {' or '.join([f'({requirement})' for requirement in self.requirements])}"


class RequiredBlock(Requirement):
    """
    This class corresponds to a list of requirements applied to agreement.
    """

    def __init__(self, record):
        self.requirements = [process_requirement(requirement) for requirement in record["requirements"]]

    def score(self, text: str, data: Dict[str, Union[int, str]], image: Image) -> Tuple[str, int]:
        result = 0
        for requirement in self.requirements:
            text, requirement_score = requirement.score(text, data, image)
            result += requirement_score
        return text, result

    def compare(self, text: str, data: Dict[str, Union[int, str]], image: Image) -> Tuple[str, bool]:
        for requirement in self.requirements:
            text, requirement_passed = requirement.compare(text, data, image)
            # print(f'block({requirement}) passed: {requirement_passed}')
            if not requirement_passed:
                return text, False
        return text, True


def process_requirement(record) -> Requirement:
    return {
        "text": RequiredText,
        "form": RequiredForm,
        "block": RequiredBlock,
        "option": RequiredOption
    }[record["type"]](record)


def load_requirements(file_name: str) -> Dict[str, Requirement]:
    with open(file_name, 'r') as file:
        forms_dict = json.load(file)
    result = dict()
    for form_name, form_data in forms_dict.items():
        result[form_name] = process_requirement(form_data)
    return result


def process_image(file_name: str, form: Requirement) -> Tuple[str, bool]:
    image = cv2.imread(file_name)
    # image = cv2.resize(image, (1650, 2340))
    # image = cv2.medianBlur(image, 3)
    # image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    image = finder.get_leveled_document(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    temporary_file_name = '.temp.png'
    cv2.imwrite(temporary_file_name, image)
    ocr_image = Image.open(temporary_file_name)
    # os.remove(temporary_file_name)
    data = tes.image_to_data(ocr_image, lang='rus', output_type=tes.Output.DICT)
    text = prepare(''.join(data['text']))
    return form.compare(text, data, ocr_image)


def test():
    requirements = load_requirements('./onti/forms.json')
    print(process_image('test2.png', requirements["form1"]))


def main():
    file_name = sys.argv[1]
    requirements = load_requirements('./onti/forms.json')
    remaining_text, is_agreement = process_image(file_name, requirements["form1"])
    print('ok' if is_agreement else '')


if __name__ == '__main__':
    sys.argv.append("test3.jpg")
    main()
