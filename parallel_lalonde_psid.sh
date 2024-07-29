#!/bin/bash

for I in {6..49}
do
  config_file="config/train/lalonde_cps/lalonde_cps_sample${I}.json"
  sbatch myjob_lalonde_cps.sh $config_file
done