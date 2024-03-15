#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --output=experiments/results/output_%j.txt
#SBATCH --error=experiments/results/error_%j.txt
#SBATCH -c 20
#SBATCH -t 1:00:00
#SBATCH -p gpu_beam
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
# You can change hostname to any command you would like to run
hostname

# Add your directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/n/data2/hms/dbmi/beamlab/manqing/DAGawareTransformer_NeurIPS"
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.3,max_split_size_mb:25


#python3 -m venv myenv
source myenv/bin/activate
module load gcc/9.2.0
module load python/3.10.11
module load cuda/12.1
#pip3 install GPUtil
#pip3 install numba
#pip3 install numba
#pip3 uninstall torch
#pip3 install torch==1.10.2
#pip3 install numpy
#pip3 install pandas
#pip3 install scikit-learn
#pip3 install wandb
#pip3 install openpyxl
#pip3 install matplotlib seaborn
#python3 summary_statistics_realcause.py
#python3 experiments/experiment_g_formula_fullsample_AIPW.py
#python3 src/data/DGP_U10.py
#python3 experiments/experiment_IPTW_unmeasuredU.py
python3 test.py
