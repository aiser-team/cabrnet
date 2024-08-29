#!/usr/bin/env bash

# Find absolute location of this script and move to repository root
location_abs=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "${location_abs}"/../../ || exit

if [ ! -d tests/compatibility/prototree/legacy ]; then
  echo "Fetching ProtoTree repository"
  # Fetch ProtoTree repository
  git clone https://github.com/M-Nauta/ProtoTree tests/compatibility/prototree/legacy
  rm -rf tests/compatibility/prototree/legacy/.git
  # Convert everything to UNIX format
  { find tests/compatibility/prototree/legacy -type f -print0 | xargs -0 dos2unix 2>&1; }> /dev/null
  # Apply minor patch to improve reproducibility and ensure compatibility
  patch -p1 -d tests/compatibility/prototree/legacy < tests/compatibility/prototree/prototree.patch
fi

if [ ! -d tests/compatibility/protopnet/legacy ]; then
  echo "Fetching ProtoPNet repository"
  # Fetch ProtoPNet repository
  git clone https://github.com/cfchen-duke/ProtoPNet tests/compatibility/protopnet/legacy
  rm -rf tests/compatibility/protopnet/legacy/.git
  # Convert everything to UNIX format
  { find tests/compatibility/protopnet/legacy -type f -print0 | xargs -0 dos2unix 2>&1; } > /dev/null
  # Apply minor patch to improve reproducibility and ensure compatibility
  patch -p1 -d tests/compatibility/protopnet/legacy < tests/compatibility/protopnet/protopnet.patch
fi

python3 tools/download_examples.py -t legacy_models