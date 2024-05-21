#!/usr/bin/env sh
python3 -m pip install loguru
python3 utils/check_docstrings.py -d src
