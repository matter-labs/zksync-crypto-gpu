#!/bin/bash

# Configuration: adjust these variables as needed
BASE_URL="https://aztec-ignition.s3.eu-west-2.amazonaws.com/MAIN+IGNITION/sealed/transcript"  # Base URL before the index
FILE_EXT=".dat"                              # File extension (if any)
START_INDEX=0                                # Starting index
END_INDEX=15                                 # Ending index

# Loop over the desired range of indexes
for ((i=START_INDEX; i<=END_INDEX; i++)); do
  # Format the index as two digits (e.g., 01, 02, ..., 10)
  FORMATTED_INDEX=$(printf "%02d" "$i")

  # Construct the full URL for the current file using the formatted index
  FILE_URL="${BASE_URL}${FORMATTED_INDEX}${FILE_EXT}"
  
  echo "Downloading ${FILE_URL} ..."
  
  # Download the file using wget
  wget "$FILE_URL"
  
  # Check if the download succeeded; if not, optionally handle errors
  if [[ $? -ne 0 ]]; then
    echo "Failed to download ${FILE_URL}"
    # Optionally break or continue based on your needs
  fi
done

echo "Download complete."