#!/usr/bin/env bash
set -euo pipefail

usage='Usage: bash scripts/train.sh CONFIG [--resume]'
case "${1:-}" in
    -h|--help)
        printf '%s\n' "$usage"
        exit 0
        ;;
    '')
        printf '%s\n' "$usage" >&2
        exit 2
        ;;
esac

config_path="$1"
shift
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ ! -f "$config_path" ]]; then
    printf 'Training config not found: %s\n' "$config_path" >&2
    exit 2
fi

export PYTHONPATH="$PWD/build/minizero/go${PYTHONPATH:+:$PYTHONPATH}"
exec python -m scripts.train --config "$config_path" "$@"
