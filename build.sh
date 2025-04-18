#!/bin/bash
rm -rf build dist .eggs
rm -rf ~/.cache/text_corrector
python3 setup.py py2app -A