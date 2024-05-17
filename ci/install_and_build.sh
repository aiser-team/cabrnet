#!/usr/bin/env sh
python3 -m pip install build
python3 -m build
python3 -m pip install -e .
