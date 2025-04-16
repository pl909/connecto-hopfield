#!/bin/bash

# --- Configuration ---
PIPELINE="ccs"           # Using CCS pipeline
STRATEGY="filt_noglobal" # Using filtered, no global signal regression strategy
DERIVATIVE="rois_cc200"  # Using CC200 ATLAS DERIVATIVE
EXT="1D"                 # File extension for ROI timeseries

# Define Sites to INCLUDE (COMMENTED OUT to download all)
# INCLUDED_SITES=(
#     "LEUVEN_1" "LEUVEN_2" "MAX_MUN" "NYU" "OHSU" "PITT" "SDSU"
#     "STANFORD" "TRINITY" "UCLA_1" "UCLA_2" "UM_1" "UM_2" "USM" "YALE"
# )

# --- Paths ---
RESULTS_BASE_DIR="results" # Where downloads will go
# Path to the full phenotypic CSV file
PHENOTYPIC_CSV="Phenotypic_V1_0b_preprocessed1.csv" # Make sure this filename is correct
DOWNLOAD_DIR="${RESULTS_BASE_DIR}/abide_timeseries/${PIPELINE}_${STRATEGY}_${DERIVATIVE}" # Download dir reflects the derivative
BASE_URL="https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs"
# --- End Configuration ---

# --- Helper function to check if site should be included (COMMENTED OUT) ---
# is_included_site() {
#   local site_to_check=$1
#   for included_site in "${INCLUDED_SITES[@]}"; do
#     if [[ "$site_to_check" == "$included_site" ]]; then
#       return 0 # 0 means true (found in included list)
#     fi
#   done
#   return 1 # 1 means false (not found in included list)
# }

# --- Main Script ---
mkdir -p "${DOWNLOAD_DIR}"

echo "Starting download to ${DOWNLOAD_DIR}..."
echo "Using derivative: ${DERIVATIVE}"
echo "Attempting to download for ALL sites listed in ${PHENOTYPIC_CSV}..."
# echo "Including ONLY sites: ${INCLUDED_SITES[*]}" # Commented out

if [ ! -f "$PHENOTYPIC_CSV" ]; then
    echo "ERROR: Phenotypic CSV file not found at $PHENOTYPIC_CSV"
    exit 1
fi

# Read the CSV file (skipping header) and download directly
# Adjust field numbers ($3=SUB_ID, $6=SITE_ID, $7=FILE_ID) if needed
# Ensure robust reading, handling potential empty fields
awk -F',' 'NR > 1 {
    # Clean fields: Remove quotes and leading/trailing whitespace
    for (i=1; i<=NF; i++) {
        gsub(/^"|"$/, "", $i);
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i);
    }
    # Print tab-separated values for easier read in bash
    print $3"\t"$6"\t"$7
}' "$PHENOTYPIC_CSV" | while IFS=$'\t' read -r sub_id site_id file_id; do

    # Basic checks for essential info
    if [[ -z "$sub_id" || -z "$site_id" ]]; then
        # echo "Skipping row: Missing SUB_ID or SITE_ID." # Optional: for debugging which rows are skipped
        continue
    fi

    # --- SITE CHECK COMMENTED OUT ---
    # Check if the site is in the INCLUDED list
    # if ! is_included_site "$site_id"; then
        # # echo "Skipping ${sub_id}: Site ${site_id} is not in the included list."
        # continue
    # fi
    # --- -------------------------- ---

    # Check if FILE_ID is valid (corresponds to subject usually)
    if [[ "$file_id" == "no_filename" || -z "$file_id" ]]; then
        echo "Skipping ${sub_id}: Invalid FILE_ID ('${file_id}')."
        continue
    fi

    # Construct URL and download path using the chosen derivative (rois_cc200)
    # Use the FILE_ID from the CSV which typically matches the needed filename part
    FILENAME="${file_id}_${DERIVATIVE}.${EXT}"
    URL="${BASE_URL}/${PIPELINE}/${STRATEGY}/${DERIVATIVE}/${FILENAME}"
    OUTPUT_PATH="${DOWNLOAD_DIR}/${FILENAME}"

    # Download if file doesn't exist
    if [ -f "$OUTPUT_PATH" ]; then
        # echo "Skipping ${FILENAME}: File already exists." # Reduce verbosity
        : # No-op, just skip
    else
        echo "Downloading ${FILENAME} (Subject: ${sub_id}, Site: ${site_id})..."
        # Use wget with flags: -q (quiet), -O (output file), --timeout, --tries
        wget --timeout=30 --tries=2 -q -O "${OUTPUT_PATH}" "${URL}"
        # Check wget exit status
        if [ $? -ne 0 ]; then
            echo "WARNING: Failed to download ${FILENAME}. URL [${URL}] might be incorrect or file unavailable."
            # Attempt to remove potentially incomplete file
            rm -f "${OUTPUT_PATH}"
        # else
            # echo "Downloaded ${FILENAME}." # Reduce verbosity further
        fi
        # Add a small sleep to be nice to the server
        sleep 0.05 # Slightly reduced sleep
    fi

done

echo "Download process finished."
echo "Files downloaded (or skipped if existing) to ${DOWNLOAD_DIR}"
echo "IMPORTANT: Ensure 'num_regions: 200' is set in your config.yaml for the CC200 atlas."