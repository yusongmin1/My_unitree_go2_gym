#!/bin/bash
# Fix mjpython on macOS by creating a symlink to libpython.
# This is needed because mjpython expects libpython in .venv/lib/

set -e

VENV_DIR=".venv"

if [[ "$(uname)" != "Darwin" ]]; then
    echo "This script is only needed on macOS."
    exit 0
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Error: .venv directory not found. Run 'uv sync' first."
    exit 1
fi

# Get Python version and prefix from the venv.
PYTHON_VERSION=$("$VENV_DIR/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_PREFIX=$("$VENV_DIR/bin/python" -c "import sys; print(sys.base_prefix)")
DYLIB_NAME="libpython${PYTHON_VERSION}.dylib"

# Find the dylib in the Python installation.
DYLIB_PATH="$PYTHON_PREFIX/lib/$DYLIB_NAME"

if [[ ! -f "$DYLIB_PATH" ]]; then
    echo "Error: Could not find $DYLIB_PATH"
    exit 1
fi

# Create the symlink.
mkdir -p "$VENV_DIR/lib"
ln -sf "$DYLIB_PATH" "$VENV_DIR/lib/$DYLIB_NAME"

echo "Created symlink: $VENV_DIR/lib/$DYLIB_NAME -> $DYLIB_PATH"
