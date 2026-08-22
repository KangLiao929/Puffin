#!/bin/bash
# Batch-download the Google Drive files listed in FILE_URLS below
# through the Drive v3 API, using aria2c for multi-connection downloads.
#
# Requires: aria2c, jq, curl, and an OAuth2 access token with Drive scope
# (tokens usually expire after ~1 hour; refresh when it does).
#
# Usage:
#   GDRIVE_ACCESS_TOKEN=ya29.xxxx bash dataset/preprocessing/drive_download.sh
#   OUTPUT_DIR=/path/out GDRIVE_ACCESS_TOKEN=... \
#       bash dataset/preprocessing/drive_download.sh

set -uo pipefail

# ================= Config =================
# Access token (from env; do NOT hardcode credentials here)
ACCESS_TOKEN="${GDRIVE_ACCESS_TOKEN:-}"

# Google Drive links to download, one per line (add more below)
FILE_URLS=(
    "https://drive.google.com/file/d/<FILE_ID_1>/view?usp=sharing"
    "https://drive.google.com/file/d/<FILE_ID_2>/view?usp=sharing"
)

# Download destination
OUTPUT_DIR="${OUTPUT_DIR:-/data/kliao/data/360_dataset}"

# ================= Dependency checks =================
if [[ -z "$ACCESS_TOKEN" ]]; then
    echo "Error: GDRIVE_ACCESS_TOKEN is not set."
    echo "Get a Drive-scope token (e.g. https://developers.google.com/oauthplayground),"
    echo "then run: GDRIVE_ACCESS_TOKEN=ya29.xxxx bash $0"
    exit 1
fi

if ! command -v aria2c &> /dev/null; then
    echo "Error: aria2c not found. Install it with 'sudo apt install aria2'."
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "Error: jq not found. Install it with 'sudo apt install jq'."
    exit 1
fi

# ================= Path checks =================
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Starting batch download"
echo "Files      : ${#FILE_URLS[@]}"
echo "Output dir : $OUTPUT_DIR"
echo "Downloader : aria2c (16 connections)"
echo "=========================================="

n_ok=0; n_fail=0; n_skip=0

# ================= Main loop =================
for URL in "${FILE_URLS[@]}"; do
    # 1. Extract the file ID (supports .../d/<id>/... and ...?id=<id> links)
    FILE_ID=$(echo "$URL" | grep -oP 'd/\K[^/?]+' || true)
    if [[ -z "$FILE_ID" ]]; then
        FILE_ID=$(echo "$URL" | grep -oP '[?&]id=\K[^&]+' || true)
    fi
    if [[ -z "$FILE_ID" ]]; then
        echo "[skip] Cannot parse a file ID from URL: $URL"
        n_skip=$((n_skip + 1))
        continue
    fi

    # 2. Resolve the original filename via the Drive API (tiny request)
    echo -n "[info] ID: $FILE_ID ... fetching metadata ... "
    FILE_METADATA=$(curl -s -H "Authorization: Bearer $ACCESS_TOKEN" \
        "https://www.googleapis.com/drive/v3/files/$FILE_ID?fields=name")
    ORIGINAL_NAME=$(echo "$FILE_METADATA" | jq -r '.name // empty')

    if [[ -z "$ORIGINAL_NAME" ]]; then
        echo "failed to resolve the filename (token expired or no permission?)."
        echo " -> Falling back to the file ID as the filename."
        ORIGINAL_NAME="${FILE_ID}.bin"
    else
        echo "name: $ORIGINAL_NAME"
    fi

    # 3. Skip finished files (a partial download leaves a .aria2 control file,
    #    in which case aria2c -c resumes it below)
    if [[ -f "$OUTPUT_DIR/$ORIGINAL_NAME" && ! -f "$OUTPUT_DIR/$ORIGINAL_NAME.aria2" ]]; then
        echo "[skip] Already downloaded: $ORIGINAL_NAME"
        n_skip=$((n_skip + 1))
        continue
    fi

    # 4. Multi-connection download
    #    -x/-s 16: 16 connections / 16 splits; -c: resume partial downloads
    if aria2c -x 16 -s 16 -c \
              --console-log-level=warn \
              --summary-interval=0 \
              --header="Authorization: Bearer $ACCESS_TOKEN" \
              "https://www.googleapis.com/drive/v3/files/$FILE_ID?alt=media" \
              -d "$OUTPUT_DIR" \
              -o "$ORIGINAL_NAME"; then
        echo "[OK] Downloaded: $ORIGINAL_NAME"
        n_ok=$((n_ok + 1))
    else
        echo "[FAILED] Download failed: $FILE_ID"
        n_fail=$((n_fail + 1))
    fi
    echo "------------------------------------------"

done

echo "All done. ok: $n_ok, failed: $n_fail, skipped: $n_skip"
