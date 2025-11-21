#!/bin/bash

# Config
BASE_URL="https://huggingface.co/datasets/sjtu-deepvision/DRR_data/resolve/main"
# BASE_URL="https://hf-mirror.com/datasets/sjtu-deepvision/DRR_data/resolve/main"
PART_PREFIX="drr.tar.part"
START=0
END=125
OUTPUT_DIR="./DRR"

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR" || exit 1

echo "Downloading parts..."

# Download all parts
for i in $(seq -f "%03g" $START $END); do
    FILE="${PART_PREFIX}${i}"
    URL="${BASE_URL}/${FILE}"

    echo "Downloading $FILE ..."
    wget -c "$URL" -O "$FILE"
done

echo "Merging parts into drr.tar ..."
cat ${PART_PREFIX}* > drr.tar

echo "Extracting drr.tar ..."
tar -xf drr.tar

echo "Done."
