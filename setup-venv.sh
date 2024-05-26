#!/usr/bin/env bash
set -uex
cd $(dirname "$0")
micromamba env create -f conda.yaml || true

if [[ ! -d venv ]]; then
  micromamba run -n mit-py311 python3 -mvenv --system-site-packages ./venv
fi
