#!/usr/bin/env bash
# The macOS PyG extension wheels (libpyg.so etc.) link against the python.org framework
# (/Library/Frameworks/Python.framework). With Homebrew Python, re-point them at Homebrew's
# framework and ad-hoc re-sign. Not needed on Linux or with python.org Python.
set -euo pipefail
VENV=${1:-.venv}
PYVER=$("$VENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
OLD=/Library/Frameworks/Python.framework/Versions/$PYVER/Python
NEW=$(brew --prefix)/opt/python@$PYVER/Frameworks/Python.framework/Versions/$PYVER/Python
SP=$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')
for f in $(find "$SP" -maxdepth 3 -name "*.so" | grep -E "libpyg|torch_sparse|torch_cluster|torch_scatter"); do
  if otool -L "$f" | grep -q "$OLD"; then
    install_name_tool -change "$OLD" "$NEW" "$f" && codesign -f -s - "$f" && echo "patched $f"
  fi
done
"$VENV/bin/python" -c "import torch_geometric.typing as t; print('pyg_lib', t.WITH_PYG_LIB, 'torch_sparse', t.WITH_TORCH_SPARSE)"
