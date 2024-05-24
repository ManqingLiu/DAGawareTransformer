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
#python3 summary_statistics_realcause.py
#python3 experiments/experiment_g_formula_fullsample_AIPW.py
#python3 experiments/train/train_cps_sample0.py
#python3 experiments/predict/predict_cps_sample0.py
#python3 experiments/train/fine_tune.py
#python3 src/data/DGP_U10.py
#python3 src/models/logistic_regression.py
#for i in {0..99}
#do
#  python3 src/doubly_robust_baseline.py \
#          --data_name \
#          lalonde_psid \
#          --data_file \
#          data/realcause_datasets/lalonde_psid/sample${i}/lalonde_psid_sample${i}.csv \
#          --results \
#          experiments/results/lalonde_psid/sample${i}/lalonde_psid.json
#done
# Add the command to run train.py with arguments
#for i in {0..99}
#do
#    python3 src/dataset.py \
#    --dag config/dag/lalonde_psid_dag.json \
#    --data_file data/realcause_datasets/lalonde_psid/sample${i}/lalonde_psid_sample${i}.csv \
#    --train_output_file data/realcause_datasets/lalonde_psid/sample${i}/train/lalonde_psid_train.csv \
#    --holdout_output_file data/realcause_datasets/lalonde_psid/sample${i}/holdout/lalonde_psid_holdout.csv
#done
#for i in {0..99}
#do
#  python3 experiments/tuning/fine_tune_psid.py \
#          --dag \
#          config/dag/lalonde_psid_dag.json \
#          --data_train_file \
#          data/realcause_datasets/lalonde_psid/sample${i}/train/lalonde_psid_train.csv \
#          --data_holdout_file \
#          data/realcause_datasets/lalonde_psid/sample${i}/holdout/lalonde_psid_holdout.csv \
#          --results \
#          config/train/lalonde_psid/sample${i}/lalonde_psid.json
#done
#for i in {0..99}
#do
#  python3 src/train.py \
#          --dag \
#          config/dag/lalonde_psid_dag.json \
#          --config \
#          config/train/lalonde_psid/sample${i}/lalonde_psid.json \
#          --mask \
#          False \
#          --data_train_file \
#          data/realcause_datasets/lalonde_psid/sample${i}/train/lalonde_psid_train.csv \
#          --data_holdout_file \
#          data/realcause_datasets/lalonde_psid/sample${i}/holdout/lalonde_psid_holdout.csv \
#          --model_train_file \
#          experiments/model/lalonde_psid/sample${i}/model_psid_train_nomask.pth \
#          --model_holdout_file \
#          experiments/model/lalonde_psid/sample${i}/model_psid_holdout_nomask.pth
#done
#for i in {0..99}
#do
#python3 src/predict.py \
#        --dag \
#        config/dag/lalonde_psid_dag.json \
#        --config \
#        config/train/lalonde_psid/sample${i}/lalonde_psid.json \
#        --mask \
#        False \
#        --data_train_file \
#        data/realcause_datasets/lalonde_psid/sample${i}/train/lalonde_psid_train.csv \
#        --data_holdout_file \
#        data/realcause_datasets/lalonde_psid/sample${i}/holdout/lalonde_psid_holdout.csv \
#        --model_train_file \
#        experiments/model/lalonde_psid/sample${i}/model_psid_train_nomask.pth \
#        --model_holdout_file \
#        experiments/model/lalonde_psid/sample${i}/model_psid_holdout_nomask.pth \
#       --output_file \
#        experiments/predict/lalonde_psid/sample${i}/predictions_psid_nomask.csv
#done
#for i in {0..99}
#do
#python3 src/evaluate.py \
#        --dag \
#        config/dag/lalonde_psid_dag.json \
#        --config \
#        config/train/lalonde_psid.json \
#        --data_file \
#        data/realcause_datasets/lalonde_psid/sample${i}/lalonde_psid_sample${i}.csv \
#        --predictions_file \
#        experiments/predict/lalonde_psid/sample${i}/predictions_psid_nomask.csv \
#        --results \
#        experiments/results/lalonde_psid/sample${i}/lalonde_psid_nomask.json
#done
python3 src/evaluate_final.py \
        --data_name \
        lalonde_psid \
        --results \
        experiments/results/lalonde_psid/lalonde_psid.json