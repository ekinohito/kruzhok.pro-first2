#!/usr/bin/env bash

python3 -m venv .
source bin/activate
sudo apt install tesseract-ocr
pip3 install numpy
pip3 install opencv-python
pip3 install Pillow
pip3 install pytesseract
chmod 777 ./main.py