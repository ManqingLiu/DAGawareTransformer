#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --output=experiments/results/output_%j.txt
#SBATCH --error=experiments/results/error_%j.txt
#SBATCH -c 15
#SBATCH -t 15:00:00
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
#python3 src/data/DGP_u_simple.py
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
#python3 src/dataset_u.py \
#        --dag \
#        config/dag/u_simple_dag.json \
#        --data_file \
#        data/unmeasured_confounding/data_U_uniform_simple.csv \
#        --train_output_file \
#       data/unmeasured_confounding/train/data_U_train_simple.csv \
#        --holdout_output_file \
#        data/unmeasured_confounding/holdout/data_U_holdout_simple.csv
#python3 src/train_u.py \
#        --dag \
#        config/dag/u_dag.json \
#        --config \
#        config/train/u.json \
#        --data_train_file \
#        data/unmeasured_confounding/train/data_U_train.csv \
#        --data_holdout_file \
#        data/unmeasured_confounding/holdout/data_U_holdout.csv \
#        --model_train_file \
#        experiments/model/unmeasured_confounding/model_U_train.pth \
#        --model_holdout_file \
#       experiments/model/unmeasured_confounding/model_U_holdout.pth
#python3 src/predict_u.py \
#       --dag \
#        config/dag/u_dag.json \
#        --config \
#       config/train/u.json \
#        --data_train_file \
#       data/unmeasured_confounding/train/data_U_train.csv \
#        --data_holdout_file \
#        data/unmeasured_confounding/holdout/data_U_holdout.csv \
#        --model_train_file \
#        experiments/model/unmeasured_confounding/model_U_train.pth \
#        --model_holdout_file \
#        experiments/model/unmeasured_confounding/model_U_holdout.pth

#python3 src/evaluate_u.py \
#        --dag \
#        config/dag/u_dag.json \
#        --config \
#        config/train/u.json \
#       --data_file \
#        data/unmeasured_confounding/data_U.csv \
#       --predictions_file \
#        data/unmeasured_confounding/accepted_samples_n100.csv \
#        --results \
#        experiments/results/unmeasured_confounding/results_n100
python3 src/rejection_sampling.py \
        --dag config/dag/u_simple_dag.json \
        --config config/train/u.json \
        --data_train_file data/unmeasured_confounding/train/data_U_train_simple.csv \
        --data_holdout_file data/unmeasured_confounding/holdout/data_U_holdout_simple.csv \
        --model_train_file experiments/model/unmeasured_confounding/model_U_train_simple.pth \
        --model_holdout_file experiments/model/unmeasured_confounding/model_U_holdout_simple.pth