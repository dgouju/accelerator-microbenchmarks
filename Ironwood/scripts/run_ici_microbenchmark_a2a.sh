#!/bin/bash

# Run command: sh ./Ironwood/scripts/run_ici_microbenchmark.sh




CONFIG_NAMES='all_to_all_3d_1024 all_to_all_2d_1024 all_to_all_1d_1024'

for CONFIG in $CONFIG_NAMES
do
  # Construct the full config file path
  CONFIG_FILE="Ironwood/configs/collectives/${CONFIG}.yaml"
  
  echo "--- Starting benchmark for ${CONFIG} ---"
  
  # Run the python script and wait for it to complete
  python Ironwood/src/run_benchmark.py --config="${CONFIG_FILE}"

  wait 
  
  echo "--- Finished benchmark for ${CONFIG} ---"
done
