#!/usr/bin/env bash

# Find absolute location of this script and move to repository root
location_abs=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "${location_abs}"/../../ || exit

declare -a legacy_repos=(
  "ProtoPNet https://github.com/cfchen-duke/ProtoPNet"
  "ProtoTree https://github.com/M-Nauta/ProtoTree"
)

for entry in "${legacy_repos[@]}"; do
  arch_name=$(echo "${entry}" | awk '{split($0,a," "); print a[1]}')
  root_path="tests/compatibility/$(echo "${arch_name}" | awk '{print tolower($0)}')"
  git_path=$(echo "${entry}" | awk '{split($0,a," "); print a[2]}')

  if [ ! -d "${root_path}/legacy" ]; then
    echo "Fetching ${arch_name} repository"
    git clone "${git_path}" "${root_path}/legacy"
    rm -rf "${root_path}/legacy/.git"
    # Convert everything to UNIX format
    { find "${root_path}/legacy" -type f -print0 | xargs -0 dos2unix 2>&1; }> /dev/null
    # Apply minor patch to improve reproducibility and ensure compatibility
    patch -p1 -d "${root_path}/legacy" < "${root_path}/corrections.patch"
  fi
done

python3 tools/download_examples.py -t legacy_models