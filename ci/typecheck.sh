#!/usr/bin/env sh
python3 -m pip install types-PyYAML types-pytz pandas-stubs mypy
python3 -m mypy src/cabrnet/
