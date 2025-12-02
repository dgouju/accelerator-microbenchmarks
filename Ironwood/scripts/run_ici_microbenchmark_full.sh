#!/bin/bash

# Run command: sh ./Ironwood/scripts/run_ici_microbenchmark.sh 4x4x4 gs://results


TOPOLOGY=$1
BUCKET=$2
TIMESTAMP=$(date +%y-%m-%d_%H-%M-%S)
CONFIG_NAMES='reduce_scatter_1d reduce_scatter_2d all_gather_3d all_reduce_3d all_to_all_3d all_gather_2d all_reduce_2d all_to_all_2d all_gather_1d all_reduce_1d all_to_all_1d'

apt-get update && apt-get install -y curl gnupg apt-transport-https ca-certificates
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
apt-get update && apt-get install -y google-cloud-sdk

for CONFIG in $CONFIG_NAMES
do

  # Construct the full config file path
  CONFIG_FILE=`python Ironwood/src/collectives_configs.py --topology=${TOPOLOGY} --collective=${CONFIG} --output_path=../microbenchmarks`

  
  echo "--- Starting benchmark for ${CONFIG} ---"
  
  # Run the python script and wait for it to complete
  python Ironwood/src/run_benchmark.py --config="${CONFIG_FILE}"

  wait 
  
  echo "--- Finished benchmark for ${CONFIG} ---"
done

# If /results is mounted (through GCSFuse for example), copy the results from pod 0 to it
if [ "${JOB_COMPLETION_INDEX}" -eq "0" ] && [ -n "${BUCKET}" ]; then
  echo "--- Copying results to ${BUCKET}/${TOPOLOGY}/${TIMESTAMP}/ ---"
  gcloud storage cp -r ../microbenchmarks/* ${BUCKET}/${TOPOLOGY}/${TIMESTAMP}/
  echo "--- Copy finished ---"
fi
