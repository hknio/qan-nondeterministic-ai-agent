#!/bin/bash

# Configuration - Number of parallel processes to run
MAX_PARALLEL=4

# Set the directory containing the scope files
SCOPE_DIR="results/scope"

# Check if the directory exists
if [ ! -d "$SCOPE_DIR" ]; then
    echo "Error: Directory $SCOPE_DIR does not exist."
    exit 1
fi

# Function to run a scope
run_scope() {
    local file=$1
    # Extract the scope_name field from the YAML file
    local scope_name=$(grep "^scope_name:" "$file" | cut -d "'" -f 2)
    
    # If scope_name is empty, try extracting from filename
    if [ -z "$scope_name" ]; then
        # Extract from filename by removing path and extension
        local filename=$(basename "$file" .yaml)
        scope_name=${filename}
    fi
    
    # Skip if scope_name is still empty
    if [ -z "$scope_name" ]; then
        echo "Warning: Could not determine scope name for $file. Skipping."
        return 0
    fi
    
    # Check if results/plan/{scope} directory exists and is non-empty
    local plan_dir="results/plan/$scope_name"
    if [ -d "$plan_dir" ] && [ "$(ls -A "$plan_dir" 2>/dev/null)" ]; then
        echo "Skipping scope: $scope_name - plan directory already exists and is non-empty"
        return 0
    fi
    
    echo "Running scope: $scope_name"
    
    # Run the Python script with the extracted scope name
    python nondeterministic_agent/planning.py --scope "$scope_name"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to run scope $scope_name"
        return 1
    fi
    
    echo "Completed scope: $scope_name"
    return 0
}

# Array to store all the scope files
scope_files=()
for file in "$SCOPE_DIR"/*.yaml; do
    # Skip if file doesn't exist (when no yaml files are found)
    [ -e "$file" ] || continue
    scope_files+=("$file")
done

# Process scope files in parallel
total_scopes=${#scope_files[@]}
echo "Found $total_scopes scopes to process with max $MAX_PARALLEL parallel processes"

current=0
while [ $current -lt $total_scopes ]; do
    # Count active jobs
    running=$(jobs -p | wc -l)
    
    # Start jobs until we reach MAX_PARALLEL
    while [ $running -lt $MAX_PARALLEL ] && [ $current -lt $total_scopes ]; do
        file=${scope_files[$current]}
        run_scope "$file" &
        ((current++))
        
        # Break if we've processed all files
        [ $current -ge $total_scopes ] && break
        
        # Update count of running jobs
        running=$(jobs -p | wc -l)
    done
    
    # Wait for any job to finish before checking again
    wait -n 2>/dev/null || true
done

# Wait for all remaining jobs to complete
wait

echo "All scopes have been processed." 