import unittest
from onti.checker import load_requirements, RequiredBlock, prepare, process_image
import os
import pytesseract
from PIL import Image
# pytesseract.pytesseract.tesseract_cmd = os.path.join('C:', 'Program Files', 'Tesseract-ocr', 'tesseract.exe')
# pytesseract.pytesseract.tesseract_cmd = r'D:\tesseract\tesseract.exe'
# pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\tesseract.exe'


class Form1Test(unittest.TestCase):

    def setUp(self) -> None:
        self.req = load_requirements(os.path.join('..', 'forms.json'))['form1']

    def test_block(self):
        self.assertIsInstance(self.req, RequiredBlock)

    def test_block_compare1(self):
        im = Image.open(os.path.join('../test.png'))
        data = pytesseract.image_to_data(im, lang='rus', output_type=pytesseract.Output.DICT)
        text = prepare(''.join(data['text']))
        self.assertEqual(self.req.compare(text, data, im), ('', False))

    def test_block_compare2(self):
        im = Image.open(os.path.join('../test2.png'))
        data = pytesseract.image_to_data(im, lang='rus', output_type=pytesseract.Output.DICT)
        text = prepare(''.join(data['text']))
        self.assertEqual(self.req.compare(text, data, im), ('', False))

    def test_block_compare3(self):
        im = Image.open(os.path.join('../test3.jpg'))
        data = pytesseract.image_to_data(im, lang='rus', output_type=pytesseract.Output.DICT)
        text = prepare(''.join(data['text']))
        self.assertEqual(self.req.compare(text, data, im), ('', False))

    def test_block_compare4(self):
        self.assertEqual(process_image('../test.png', self.req), ('', False))

    def test_block_compare5(self):
        self.assertEqual(process_image('../test2.png', self.req), ('', False))

    def test_block_compare6(self):
        self.assertEqual(process_image('../test3.jpg', self.req), ('', False))


if __name__ == '__main__':
    unittest.main()
