#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# A relative source of '.' is resolved by Python in this native working directory.
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 9))' >/dev/null 2>&1; then
    exec "$candidate" publishing/publish.py "$@"
  fi
done
if command -v py >/dev/null 2>&1 && py -3 -c 'import sys; sys.exit(sys.version_info < (3, 9))' >/dev/null 2>&1; then
  exec py -3 publishing/publish.py "$@"
fi
printf '%s\n' 'A working Python 3.9+ interpreter is required (python3, python, or py -3).' >&2
exit 1
