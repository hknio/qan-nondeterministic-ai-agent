#!/bin/bash

MAX_PARALLEL=8
PLAN_DIR="results/plan"

cleanup() {
    echo
    echo "--- Caught interrupt signal. Terminating child processes... ---"
    local script_pid=$$
    echo "Script PID: $script_pid"

    echo "Sending SIGTERM to processes with PPID $script_pid..."
    pkill -TERM -P $script_pid

    sleep 1

    children_pids=$(pgrep -P $script_pid)
    if [ -n "$children_pids" ]; then
       echo "Graceful shutdown appears incomplete after delay. Sending SIGKILL..."
       pkill -KILL -P $script_pid
       sleep 1
    fi

    echo "--- Waiting for child processes to exit... ---"
    wait
    echo "--- Child processes terminated. Exiting script. ---"
    exit 130
}
trap cleanup SIGINT SIGTERM

if [ ! -d "$PLAN_DIR" ]; then
    echo "Error: Directory $PLAN_DIR does not exist."
    trap - SIGINT SIGTERM
    exit 1
fi
run_execution() {
    local scope_dir=$1
    local scope_name=$(basename "$scope_dir")

    if [ -z "$scope_name" ]; then
        echo "Warning: Could not determine scope name for $scope_dir. Skipping."
        return 0
    fi

    echo "[$$ -> $$] Starting execution for scope: $scope_name"

    python3 nondeterministic_agent/execution.py --scope "$scope_name" --language c --non-interactive

    local exit_code=$?
    if [ $exit_code -eq 130 ] || [ $exit_code -eq 143 ] ; then
         echo "[$$] Execution INTERRUPTED for scope $scope_name (Exit Code: $exit_code)"
         return 0
    elif [ $exit_code -ne 0 ]; then
        echo "[$$] Error: Failed to run execution for scope $scope_name (Exit Code: $exit_code)"
        return 1
    fi

    echo "[$$] Completed execution for scope: $scope_name"
    return 0
}
scope_dirs=()
for dir in "$PLAN_DIR"/*/; do
    # Use -d to check if it's a directory
    [ -d "$dir" ] || continue
    scope_dirs+=("$dir")
done

# Process scope directories in parallel
total_scopes=${#scope_dirs[@]}
echo "Found $total_scopes scope plans to process with max $MAX_PARALLEL parallel processes (Script PID: $$)"

current=0
while [ $current -lt $total_scopes ]; do
    # Count active jobs using pgrep for potentially more accuracy than 'jobs'
    # Count processes whose parent is the current script
    running=$(pgrep -P $$ | wc -l)

    # Start jobs until we reach MAX_PARALLEL or run out of plans
    while [ $running -lt $MAX_PARALLEL ] && [ $current -lt $total_scopes ]; do
        dir=${scope_dirs[$current]}
        # Run in background
        run_execution "$dir" &
        # Store the PID of the last background process (optional, for tracking)
        # last_pid=$!
        # echo "Launched PID $last_pid for scope $(basename $dir)"
        ((current++))

        # Update count of running jobs immediately after launching
        running=$(pgrep -P $$ | wc -l)
    done

    # Wait for *any* child process of this script to finish
    # This uses the 'wait' builtin without -n, waiting for any child,
    # not just background jobs known to the shell's job control.
    # It might block longer if non-background children exist, but is safer here.
    wait -p child_pid 2>/dev/null # Get PID of waited process if possible
    wait_exit_code=$?

    # Optional: Check exit code of the process that just finished
    # if [ $wait_exit_code -ne 0 ] && [ $wait_exit_code -ne 130 ] && [ $wait_exit_code -ne 143 ]; then
    #     echo "Warning: Child process PID $child_pid exited with code $wait_exit_code."
    #     # Decide if you want to stop the whole script on child failure
    #     # cleanup
    # fi

    # If wait failed because there are no children left (or an error), break the loop
    if [ $wait_exit_code -eq 127 ]; then
        echo "No more child processes to wait for."
        break
    fi
done

# Wait for all remaining background jobs launched by this script to complete
echo "--- Waiting for any remaining child processes to complete naturally... ---"
wait
echo "--- All executions have been processed. ---"

# Disable the trap explicitly before exiting normally
trap - SIGINT SIGTERM
exit 0