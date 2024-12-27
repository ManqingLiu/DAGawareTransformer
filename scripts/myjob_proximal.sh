#!/bin/bash
#SBATCH --job-name=myjob_array
#SBATCH --output=experiments/results/output_%A_%a.txt
#SBATCH --error=experiments/results/error_%A_%a.txt
#SBATCH -c 15
#SBATCH -t 10:00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:teslaV100s:1
#SBATCH --mem=15G
#SBATCH --array=0-19
hostname

# Add your directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/n/data2/hms/dbmi/beamlab/manqing/DAGawareTransformer"
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.3,max_split_size_mb:512


#python3 -m venv myenv
source myenv/bin/activate
module load gcc/9.2.0
module load python/3.10.11


# Use the SLURM_ARRAY_TASK_ID as the sample index
sample_index=$SLURM_ARRAY_TASK_ID

python3 src/experiment_proximal.py \
         --dag \
        config/dag/proximal_dag_z.json \
        --config \
        config/train/proximal/nmmr_v_z_transformer_n50000.json \
        --results_dir \
        experiments/results/proximal \
        --sample_index $sample_index
