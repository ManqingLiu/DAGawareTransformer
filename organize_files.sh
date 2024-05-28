#!/bin/bash
#SBATCH --job-name=organize_files
#SBATCH --output=experiments/results/output_%j.txt
#SBATCH --error=experiments/results/error_%j.txt
#SBATCH -c 15
#SBATCH -t 1:00:00
#SBATCH -p gpu_beam
#SBATCH --gres=gpu:1
#SBATCH --mem=10G

# Loop over the range from 0 to 99
for i in {0..99}
do
    # Create a directory named "sample$i"
    #mkdir "data/realcause_datasets/lalonde_psid/sample$i"
    #mkdir "experiments/model/lalonde_psid/sample$i"
    mkdir "experiments/predict/lalonde_psid"
    mkdir "experiments/predict/lalonde_psid/sample$i"

    # Move the file "lalonde_psid_sample$i" to the directory "sample$i"
    #mv "data/realcause_datasets/lalonde_psid/lalonde_psid_sample$i.csv" "data/realcause_datasets/lalonde_psid/sample$i/"

    # Create a directory named "sample$i"
    #mkdir "data/realcause_datasets/lalonde_cps/sample$i"
    #mkdir "experiments/model/lalonde_cps/sample$i"
    mkdir "experiments/predict/lalonde_cps"
    mkdir "experiments/predict/lalonde_cps/sample$i"

    # Move the file "lalonde_cps_sample$i" to the directory "sample$i"
    #mv "data/realcause_datasets/lalonde_cps/lalonde_cps_sample$i.csv" "data/realcause_datasets/lalonde_cps/sample$i/"

    # Create a directory named "sample$i"
    #mkdir "data/realcause_datasets/twins/sample$i"
    #mkdir "experiments/model/twins/sample$i"
    mkdir "experiments/predict/twins"
    mkdir "experiments/predict/twins/sample$i"

    # Move the file "twins_sample$i" to the directory "sample$i"
    #mv "data/realcause_datasets/twins/twins_sample$i.csv" "data/realcause_datasets/twins/sample$i/"

    # Create 'holdout' and 'train' directories in each 'sample' directory
    # mkdir "data/realcause_datasets/lalonde_psid/sample$i/holdout"
    # mkdir "data/realcause_datasets/lalonde_psid/sample$i/train"
    # mkdir "data/realcause_datasets/lalonde_cps/sample$i/holdout"
    # mkdir "data/realcause_datasets/lalonde_cps/sample$i/train"
    # mkdir "data/realcause_datasets/twins/sample$i/holdout"
    # mkdir "data/realcause_datasets/twins/sample$i/train"
done