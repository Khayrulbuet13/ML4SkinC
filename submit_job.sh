#!/bin/bash

# Define paths
script_path="your_script.py"

# Define maximum number of concurrent jobs
max_jobs=4

# Define a list of configurations
declare -a configurations=(
    "0.001 0.0001 64 1000 resnet18-VS_loss-SGD True 0.9 0.4 1.0"
    "0.0005 0.00005 32 500 resnet18-VS_loss-SGD False 0.8 0.3 0.9"
    # Add more configurations as needed
)

# Current number of running jobs
running_jobs=0

# Job submission function
submit_job() {
    IFS=' ' read lr wd bs e model tr omega gamma tau <<< "$1"
    python $script_path --learning-rate $lr --weight-decay $wd --batch-size $bs --epochs $e --model $model --train-resnet $tr --omega $omega --gamma $gamma --tau $tau
    echo "Job with lr=$lr, wd=$wd, bs=$bs, e=$e, model=$model, tr=$tr, omega=$omega, gamma=$gamma, tau=$tau completed."
}

# Export the function so it can be used by subshells
export -f submit_job
export script_path

# Job control loop
while [ ${#configurations[@]} -gt 0 ]; do
    if [ "$running_jobs" -lt "$max_jobs" ]; then
        # Get the first configuration from the array
        config=${configurations[0]}
        # Remove it from the array
        configurations=("${configurations[@]:1}")
        # Submit job in the background
        submit_job "$config" &
        # Increment the running jobs count
        ((running_jobs++))
        echo "Submitted job with configuration: $config. Total running jobs: $running_jobs"
    else
        # Wait for any job to finish
        wait -n
        # Decrement the running jobs count
        ((running_jobs--))
    fi
done

# Wait for all remaining jobs to complete
wait
echo "All jobs have completed."
