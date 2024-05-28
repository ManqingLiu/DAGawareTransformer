#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --output=experiments/results/output_%j.txt
#SBATCH --error=experiments/results/error_%j.txt
#SBATCH -c 20
#SBATCH -t 1:00:00
#SBATCH -p gpu_beam
#SBATCH --gres=gpu:1
#SBATCH --mem=15G
# You can change hostname to any command you would like to run
hostname

# Add your directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/n/data2/hms/dbmi/beamlab/manqing/DAGawareTransformer_NeurIPS"
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.3,max_split_size_mb:512


#python3 -m venv myenv
source myenv/bin/activate
module load gcc/9.2.0
module load python/3.10.11

python3 src/data/DGP_nhefs.py