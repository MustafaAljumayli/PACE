#!/bin/bash

# Ensure script exits on error
set -e

python -m venv .venv
source .venv/bin/activate
pip install -e .