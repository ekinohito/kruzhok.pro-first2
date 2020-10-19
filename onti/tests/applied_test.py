import unittest
from onti.checker import load_requirements, RequiredBlock
import os
import pytesseract
from PIL import Image


class Form1Test(unittest.TestCase):

    def setUp(self) -> None:
        self.req = load_requirements(os.path.join('..', 'forms.json'))['form1']

    def test_block(self):
        self.assertIsInstance(self.req, RequiredBlock)

    def test_block_compare1(self):
        im = Image.open(os.path.join('../test.png'))
        data = pytesseract.image_to_data(im, lang='rus')
        text = 'зарегистрированн'
        self.assertEqual(self.req.compare(text, data, im), (text, True))

    def test_block_compare2(self):
        im = Image.open(os.path.join('../test.png'))
        data = pytesseract.image_to_data(im, lang='rus')
        text = 'поадресу'
        self.assertEqual(self.req.compare(text, data, im), (text, True))



if __name__ == '__main__':
    unittest.main()