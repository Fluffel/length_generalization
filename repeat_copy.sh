#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./gen_sample.sh            # random length in [1, 20]
#   ./gen_sample.sh 8          # exactly 8 tokens
#   ./gen_sample.sh 5 12       # random length in [5, 12]
#
# Output:
#   <bos> [random a/b sequence] <sep> [exact copy] <eos>

min_len="${1:-1}"
max_len="${2:-20}"

# If only one arg is given, use fixed length.
if [[ $# -eq 1 ]]; then
  max_len="$min_len"
fi

if (( min_len < 1 || max_len < min_len )); then
  echo "Error: length must satisfy 1 <= min_len <= max_len" >&2
  exit 1
fi

# Random sequence length
len=$(( RANDOM % (max_len - min_len + 1) + min_len ))

# Build random a/b sequence
seq=()
for ((i=0; i<len; i++)); do
  if (( RANDOM % 2 )); then
    seq+=("a")
  else
    seq+=("b")
  fi
done

left="${seq[*]}"
right="${seq[*]}"  # exact copy

output="<bos> ${left} <sep> ${right} <eos>"

# Copy to clipboard when a clipboard utility is available.
copied=false
if command -v wl-copy >/dev/null 2>&1; then
  if printf '%s' "$output" | wl-copy >/dev/null 2>&1; then
    copied=true
  fi
fi
if ! $copied && command -v xclip >/dev/null 2>&1; then
  if printf '%s' "$output" | xclip -selection clipboard >/dev/null 2>&1; then
    copied=true
  fi
fi
if ! $copied && command -v xsel >/dev/null 2>&1; then
  if printf '%s' "$output" | xsel --clipboard --input >/dev/null 2>&1; then
    copied=true
  fi
fi
if ! $copied; then
  echo "Warning: failed to copy to clipboard (tried wl-copy, xclip, xsel)." >&2
fi

echo "$output"