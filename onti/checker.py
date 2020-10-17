import jsonimport osimport refrom abc import ABC, abstractmethodfrom pprint import pprintfrom typing import Tuple, List, Dict, Unionfrom PIL import Imageimport cv2import pytesseract as tesdef prepare(s: str):    """    produces lowercase russian-only symbols    :param s: given string    :return: prepared string    """    return re.sub(r'[^а-я0-9]', '', s.lower())class Requirement(ABC):    """    This abstract class corresponds to requirement. Any acceptable agreement should pass it's test.    """    @abstractmethod    def score(self, text: str, data: Dict[str, Union[int, str]]) -> Tuple[str, int]:        """        scores text in some way        :param text: given text        :param data: text data        :return: remaining text and score        """        pass    @abstractmethod    def compare(self, text: str, data: Dict[str, Union[int, str]]) -> Tuple[str, bool]:        """        determines whether text contains requirement or not        :param text: given text        :param data: text data        :return: True if text is acceptable, False otherwise        """        passclass RequiredText(Requirement):    """    This class corresponds to text data which should be represented in agreement.    """    def __init__(self, pattern: str, threshold: int = 0):        """        required text        :param pattern: prepared text which must be presented in agreement        :param threshold: maximum possible value of score to pass test        """        self.pattern = pattern        self.threshold = threshold    def score(self, text: str, data: Dict[str, Union[int, str]]) -> Tuple[str, int]:        match = re.search(self.pattern, text)        if match:            return text[match.regs[0][1]:], 0        return '', 100    def compare(self, text: str, data: Dict[str, Union[int, str]]) -> Tuple[str, bool]:        remaining, score = self.score(text, data)        return remaining, score <= self.threshold    def __repr__(self):        return f'text: {self.pattern}'class RequiredForm(Requirement):    """    This class corresponds to some form in agreement which must be filled.    """    def __init__(self, form_record: Dict[str, Union[int, str]]):        """        required form        :param form_record: dictionary containing anchors names and percentage of minimum possible fulfillment        """        self.left_anchor = form_record['left']        self.right_anchor = form_record['right']        self.threshold = form_record['threshold']    def score(self, text: str, data: Dict[str, Union[int, str]]) -> Tuple[str, int]:        left_index = None        for i, anchor_name in enumerate(data['text']):            if re.search(self.left_anchor, anchor_name):                left_index = i        if not left_index:            return text, 0        return text, 9000    def compare(self, text: str, data: Dict[str, Union[int, str]]) -> Tuple[str, bool]:        remaining, score = self.score(text, data)        return remaining, score >= self.thresholdclass RequiredBlock(Requirement):    """    This class corresponds to a list of requirements applied to agreement.    """    def __init__(self, blocks: List[Requirement]):        self.blocks = blocks.copy()    def score(self, text: str, data: Dict[str, Union[int, str]]) -> Tuple[str, int]:        result = 0        for block in self.blocks:            text, block_score = block.score(text, data)            result += block_score        return text, result    def compare(self, text: str, data: Dict[str, Union[int, str]]) -> Tuple[str, bool]:        for block in self.blocks:            text, block_passed = block.compare(text, data)            print(f'block({block}) passed: {block_passed}')            if not block_passed:                return text, False        return text, Truedef load_requirements(file_name: str) -> Dict[str, RequiredBlock]:    with open(file_name, 'r') as file:        requirements_dict = json.load(file)    result = dict()    for block_name, block_data in requirements_dict.items():        result[block_name] = RequiredBlock([RequiredText(record) if type(record) == str                                            else RequiredForm(record) for record in block_data])    return resultdef test():    requirements = load_requirements('forms.json')    image = cv2.imread('test.png')    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    # image = cv2.medianBlur(image, 3)    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]    temporary_file_name = '.temp.png'    cv2.imwrite(temporary_file_name, image)    ocr_image = Image.open(temporary_file_name)    data = tes.image_to_data(ocr_image, lang='rus', output_type=tes.Output.DICT)    text = prepare(''.join(data['text']))    requirements["form1"].compare(text, data)    '''    text = tes.image_to_string(ocr_image, lang='rus')    text_upside_down = tes.image_to_string(ocr_image.rotate(180))    # os.remove(temporary_file_name)    prepared_text = prepare(text)    print(prepared_text)    print(f'0deg comparing: {requirements["form1"].compare(prepared_text)}')    print('-' * 30)    prepared_text = prepare(text_upside_down)    print(prepared_text)    print(f'180deg comparing: {requirements["form1"].compare(prepared_text)}')    '''if __name__ == '__main__':    test()