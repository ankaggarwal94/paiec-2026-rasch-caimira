#!/usr/bin/env bash
# Compatibility entry point. The supported cross-platform launcher is publish.sh.
exec bash "$(dirname -- "$0")/publish.sh" "$@"
