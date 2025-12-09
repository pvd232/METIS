#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-.}"

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Directory not found: $TARGET_DIR" >&2
  exit 1
fi

# Default output goes into the target directory unless explicitly provided
TARGET_DIR_CLEAN="$(basename "$(cd "$TARGET_DIR" && pwd)")"
OUTPUT_PDF="${2:-"$TARGET_DIR/combined_python_code_${TARGET_DIR_CLEAN}.pdf"}"


TEMP_PS_FILE="$(mktemp --suffix=.ps)"

shopt -s nullglob
files=( "$TARGET_DIR"/*.py )

if ((${#files[@]} == 0)); then
  echo "No .py files found in '$TARGET_DIR'."
  exit 0
fi

# Version-aware sort by basename (config.py, config2.py, config10.py)
basenames=()
for f in "${files[@]}"; do
  basenames+=( "$(basename "$f")" )
done

IFS=$'\n' basenames_sorted=($(printf '%s\n' "${basenames[@]}" | sort -V))
unset IFS

files_sorted=()
for b in "${basenames_sorted[@]}"; do
  files_sorted+=( "$TARGET_DIR/$b" )
done

echo "Generating PostScript..."
enscript -Epython --color -q -p "$TEMP_PS_FILE" "${files_sorted[@]}"

echo "Generating PDF..."
ps2pdf "$TEMP_PS_FILE" "$OUTPUT_PDF"

rm -f "$TEMP_PS_FILE"
echo "PDF '$OUTPUT_PDF' created successfully."
