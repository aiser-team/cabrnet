#!/usr/bin/env bash

if [ ! -d src/legacy/prototree ]; then
  echo "Fetching ProtoTree repository"
  # Fetch ProtoTree repository
  git clone https://github.com/M-Nauta/ProtoTree src/legacy/prototree
  # Apply minor patch to improve reproducibility and ensure compatibility
  git apply --directory src/legacy/prototree src/legacy/patches/prototree.patch
  rm -rf src/legacy/prototree/.git
fi

if [ ! -d data/CUB_200_2011/dataset ]; then
  echo "Downloading CUB-200-2011 dataset"
  mkdir -p data/CUB_200_2011
  msg=$(python3 src/legacy/prototree/preprocess_data/download_birds.py 2>&1)
  if [ $? != 0 ]; then
    echo $msg
    exit
  fi
  # Remove attribute file
  rm data/attributes.txt
  echo "Preprocessing images (this may take a while)"
  msg=$(python3 src/legacy/prototree/preprocess_data/cub.py 1&> /dev/null)
  if [ $? != 0 ]; then
    echo $msg
    exit
  fi
fi