#!/bin/zsh

# Iterate over each *.mlpackage file in the current directory
for mlpackage in *.mlpackage; do
    # Skip if no matches found
    [[ -e "$mlpackage" ]] || continue
    # Call the hf upload command for each file
    hf upload "$mlpackage"
done
