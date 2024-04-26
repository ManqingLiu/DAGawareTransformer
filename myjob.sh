#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --output=experiments/results/output_%j.txt
#SBATCH --error=experiments/results/error_%j.txt
#SBATCH -c 15
#SBATCH -t 1:00:00
#SBATCH -p gpu_beam
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
# You can change hostname to any command you would like to run
hostname

# Add your directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/n/data2/hms/dbmi/beamlab/manqing/DAGawareTransformer_NeurIPS"
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.3,max_split_size_mb:512


#python3 -m venv myenv
source myenv/bin/activate
module load gcc/9.2.0
module load python/3.10.11
#module load cuda/12.1
#pip3 install GPUtil
#pip3 install numba
#pip3 uninstall torch
#pip3 install torch==1.10.2
#pip3 install numpy
#pip3 install pandas
#pip3 install scikit-learn
#pip3 install wandb
#pip3 install "ray[tune]"
#pip3 install openpyxl
#pip3 install matplotlib seaborn
#pip3 install doubleml
#pip3 install pqdm
python3 src/data/DGP_unmeasured_confounding.py
#python3 summary_statistics_realcause.py
#python3 experiments/experiment_g_formula_fullsample_AIPW.py
#python3 experiments/train/train_cps_sample0.py
#python3 experiments/predict/predict_cps_sample0.py
#python3 experiments/train/fine_tune.py
#python3 src/data/DGP_U10.py
#python3 src/models/logistic_regression.py
#python3 experiments/experiment_IPTW_unmeasuredU.py
#python3 src/data/data_preprocess.py
#python3 experiments/Monte-Carlo/train_cps.py
#python3 experiments/Monte-Carlo/train_psid.py
#python3 experiments/Monte-Carlo/predict_psid.py
#python3 experiments/Monte-Carlo/predict_cps.py
#python3 experiments/train/train_stratified_std.py
#python3 experiments/predict/predict_stratified_std_v2.py
# Add the command to run train.py with arguments
#python3 src/dataset.py \
#        --dag \
#        config/dag/lalonde_psid_dag.json \
#        --data_file \
#        data/realcause_datasets/lalonde_psid/sample0/lalonde_psid_sample0.csv
#python3 src/train.py \
#        --dag \
#        config/dag/lalonde_psid_dag.json \
#        --config \
#      config/train/lalonde_psid.json \
#        --data_train_file \
#        data/realcause_datasets/lalonde_psid/sample0/train/lalonde_psid_train.csv \
#        --data_holdout_file \
#       data/realcause_datasets/lalonde_psid/sample0/holdout/lalonde_psid_holdout.csv \
#        --model_train_file \
#        experiments/model/lalonde_psid/sample0/model_psid_train_nomask.pth \
#        --model_holdout_file \
#        experiments/model/lalonde_psid/sample0/model_psid_holdout_nomask.pth \
#python3 src/predict.py \
#        --dag \
#        config/dag/lalonde_psid_dag.json \
#        --config \
#       config/train/lalonde_psid.json \
#        --data_train_file \
#       data/realcause_datasets/lalonde_psid/sample0/train/lalonde_psid_train.csv \
#        --data_holdout_file \
#        data/realcause_datasets/lalonde_psid/sample0/holdout/lalonde_psid_holdout.csv \
#        --model_train_file \
#        experiments/model/lalonde_psid/sample0/model_psid_train_nomask.pth \
#       --model_holdout_file \
#        experiments/model/lalonde_psid/sample0/model_psid_holdout_nomask.pth

#python3 src/evaluate.py \
#        --dag \
#        config/dag/lalonde_psid_dag.json \
#        --config \
#        config/train/lalonde_psid.json \
#       --data_file \
#      data/realcause_datasets/lalonde_psid/sample0/lalonde_psid_sample0.csv \
#       --predictions_file \
#        experiments/predict/lalonde_psid/sample0/predictions_psid.csv