#!/usr/bin/env bash

if [ ! -d src/legacy/prototree ]; then
  # Fetch ProtoTree repository
  git clone https://github.com/M-Nauta/ProtoTree src/legacy/prototree
  # Apply minor patch to improve reproducibility and ensure compatibility
  git apply --directory src/legacy/prototree src/legacy/patches/prototree.patch
  rm -rf src/legacy/prototree/.git
fi