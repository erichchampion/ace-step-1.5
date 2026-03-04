#!/bin/zsh

# Increase timeout for large file uploads (5 minutes)
export HF_HUB_HTTP_TIMEOUT=300

# Iterate over each *.mlpackage file in the current directory
for mlpackage in *.mlpackage; do
    # Skip if no matches found
    [[ -e "$mlpackage" ]] || continue
    
    echo "--- Uploading $mlpackage ---"
    
    # Retry logic (up to 3 attempts)
    for attempt in {1..3}; do
        echo "Attempt $attempt..."
        if hf upload "$mlpackage"; then
            echo "Successfully uploaded $mlpackage"
            break
        else
            echo "Error uploading $mlpackage (Attempt $attempt failed)"
            if [[ $attempt -lt 3 ]]; then
                echo "Retrying in 10 seconds..."
                sleep 10
            else
                echo "Failed to upload $mlpackage after 3 attempts."
            fi
        fi
    done
done
