#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Every test_*.py runs, so a new test file can't be forgotten. A file with no assertions
# (e.g. empty) would "pass" without testing anything, so it fails the run instead.
# Tests are either unittest classes or plain scripts asserting in main(): both assert.
shopt -s nullglob
tests=(test_*.py)
if (( ${#tests[@]} == 0 )); then
  echo "No test_*.py files found" >&2
  exit 1
fi
for t in "${tests[@]}"; do
  if [[ ! -s "$t" ]] || ! grep -qE '\bassert' "$t"; then
    echo "FAIL: $t is empty or contains no assertions" >&2
    exit 1
  fi
done
for t in "${tests[@]}"; do
  echo "== $t"
  python "$t"
done
