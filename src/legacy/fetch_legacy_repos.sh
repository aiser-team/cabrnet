#!/usr/bin/env bash

# Find absolute location of this script and move to repository root
location_abs=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "${location_abs}"/../../ || exit

if [ ! -d src/legacy/prototree ]; then
  echo "Fetching ProtoTree repository"
  # Fetch ProtoTree repository
  git clone https://github.com/M-Nauta/ProtoTree src/legacy/prototree
  rm -rf src/legacy/prototree/.git
  # Convert everything to UNIX format
  { find src/legacy/prototree -type f -print0 | xargs -0 dos2unix 2>&1; }> /dev/null
  # Apply minor patch to improve reproducibility and ensure compatibility
  patch -p1 -d src/legacy/prototree < src/legacy/patches/prototree.patch
fi

if [ ! -d src/legacy/protopnet ]; then
  echo "Fetching ProtoPNet repository"
  # Fetch ProtoPNet repository
  git clone https://github.com/cfchen-duke/ProtoPNet src/legacy/protopnet
  rm -rf src/legacy/protopnet/.git
  # Convert everything to UNIX format
  { find src/legacy/protopnet -type f -print0 | xargs -0 dos2unix 2>&1; } > /dev/null
  # Apply minor patch to improve reproducibility and ensure compatibility
  patch -p1 -d src/legacy/protopnet < src/legacy/patches/protopnet.patch
fi

if [ ! -d data/CUB_200_2011 ]; then
  echo "Downloading CUB-200-2011 dataset"
  mkdir -p data/CUB_200_2011
  if ! msg=$(python3 src/legacy/prototree/preprocess_data/download_birds.py 2>&1); then
    echo "FAILED: ${msg}"
    exit
  fi
  # Remove attribute file
  rm data/attributes.txt
fi

if [ ! -d data/CUB_200_2011/dataset/train_corners ]; then
  echo "Performing ProtoTree augmentation (this may take a while)"
  if ! msg=$(python3 src/legacy/prototree/preprocess_data/cub.py 1&> /dev/null); then
    echo "FAILED: ${msg}"
    exit
  fi
fi

if [ ! -d data/CUB_200_2011/dataset/train_crop_augmented ]; then
  echo "Performing ProtoPNet augmentation (this may take a while)"
  #if ! msg=$(python3 src/legacy/protopnet/img_aug.py 1&> /dev/null); then
  if ! msg=$(python3 src/legacy/protopnet/img_aug.py); then
    echo "FAILED: ${msg}"
    exit
  fi
fi