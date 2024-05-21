#!/usr/bin/env sh
python3 -m pip install black
python3 -m black --check src tools utils
