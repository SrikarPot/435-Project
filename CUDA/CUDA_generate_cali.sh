#!/bin/bash

algorithms=("merge" "enumeration" "bitonic" "oddeven")
num_processes=(2 4 8 16 32 64 128 256 512 1024)
num_vals=(65536 262144 1048576 4194304 16777216 67108864 268435456)
input_types=("Sorted" "Random" "ReverseSorted" "1%perturbed")

# Iterate over all combinations
for algorithm in "${algorithms[@]}"; do
  for input_type in "${input_types[@]}"; do
    for val_count in "${num_vals[@]}"; do
      for process_count in "${num_processes[@]}"; do
        # Formulate the sbatch command
        sbatch_command="sbatch cuda.grace_job $algorithm $process_count $val_count $input_type"
        
        # Print the command before executing (optional)
        echo "Running command: $sbatch_command"
        
        # Execute the sbatch command
        eval $sbatch_command
      done
    done
  done
done

# chmod +x CUDA_generate_cali.sh
# ./CUDA_generate_cali.sh