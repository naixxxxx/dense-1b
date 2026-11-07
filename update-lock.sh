#!/usr/bin/env bash
set -euo pipefail

# Run this from the dense-1b Git working directory.
if [[ ! -f pyproject.toml ]]; then
  echo 'Please cd to the dense-1b directory before running this script.' >&2
  exit 1
fi

# Keep the previous lock as a local safety copy.
if [[ -f uv.lock ]]; then
  cp -f uv.lock uv.lock.before-cu128.bak
fi

# A Python 3.12 interpreter is needed by the project.
uv python install 3.12
printf '3.12\n' > .python-version

# Re-resolve against the explicit cu128 PyTorch index. No packages are installed here.
uv lock --upgrade-package torch --upgrade-package triton

# Check what was resolved before sync/install.
uv tree --depth 1
printf '\nLock regenerated. Inspect torch, triton and CUDA 12.x entries in uv.lock.\n'
