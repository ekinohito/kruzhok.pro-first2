#!/usr/bin/env bash

mkdir venv
cd ./venv || exit
python3 -m venv .
source bin/activate
cd ../
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-rus
pip3 install numpy
pip3 install opencv-python
pip3 install Pillow
pip3 install pytesseract
chmod 777 ./main.py