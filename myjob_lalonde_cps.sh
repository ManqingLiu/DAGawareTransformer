#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --output=experiments/results/output_%j.txt
#SBATCH --error=experiments/results/error_%j.txt
#SBATCH -c 10
#SBATCH -t 24:00:00
#SBATCH -p gpu_beam
#SBATCH --gres=gpu:4
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
#python3 experiments/tuning/fine_tune_cps.py \
#        --config config/train/lalonde_cps.json
#python3 experiments/tuning/fine_tune_cps_cfcv.py \
#        --config config/train/lalonde_cps.json
#python3 experiments/tuning/fine_tune_psid_ipw.py \
#for i in {0..49}
#do
#  python3 experiments/tuning/fine_tune.py \
#          --config \
#          config/train/lalonde_cps/lalonde_cps_sample${i}.json \
#          --estimator \
#          cfcv \
#          --data_name \
#          lalonde_cps
#done
#python3 experiments/tuning/fine_tune_cps_cfcv.py --config $1
#python3 src/experiment.py \
#        --config config/train/lalonde_cps/lalonde_cps_sample1.json \
#        --estimator \
#        cfcv \
#        --data_name \
#        lalonde_cps
for i in {0..29}
do
python3 src/experiment.py \
        --config config/train/lalonde_cps/lalonde_cps_sample${i}.json \
        --estimator cfcv \
        --data_name lalonde_cps
done
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
#for i in {0..99}
#do
#  python3 src/doubly_robust_baseline.py \
#          --data_name \
#          lalonde_cps \
#          --data_file \
#          data/realcause_datasets/lalonde_cps/sample${i}/lalonde_cps_sample${i}.csv \
#          --results \
#          experiments/results/lalonde_cps/sample${i}/lalonde_cps.json
#done
# Add the command to run train.py with arguments
#for i in {0..99}
#do
#    python3 src/dataset.py \
#    --dag config/dag/lalonde_cps_dag.json \
#    --data_file data/realcause_datasets/lalonde_cps/sample${i}/lalonde_cps_sample${i}.csv \
#    --train_output_file data/realcause_datasets/lalonde_cps/sample${i}/train/lalonde_cps_train.csv \
#    --holdout_output_file data/realcause_datasets/lalonde_cps/sample${i}/holdout/lalonde_cps_holdout.csv
#done
#for i in {0..99}
#do
#python3 experiments/tuning/fine_tune_cps.py \
#          --dag \
#          config/dag/lalonde_cps_dag.json \
#          --data_train_file \
#          data/realcause_datasets/lalonde_cps/sample${i}/train/lalonde_cps_train.csv \
#          --data_holdout_file \
#          data/realcause_datasets/lalonde_cps/sample${i}/holdout/lalonde_cps_holdout.csv \
#          --results \
#          config/train/lalonde_cps/sample${i}/lalonde_cps.json
#done
#python3 src/dataset.py \
#        --dag \
#        config/dag/lalonde_cps_dag.json \
#        --data_file \
#        data/realcause_datasets/lalonde_cps/sample38/lalonde_cps_sample38.csv \
#        --train_output_file \
#        data/realcause_datasets/lalonde_cps/sample38/train/lalonde_cps_train.csv \
#        --holdout_output_file \
#        data/realcause_datasets/lalonde_cps/sample38/holdout/lalonde_cps_holdout.csv
#for i in {0..99}
#do
#python3 src/train.py \
#        --dag \
#        config/dag/lalonde_cps_dag.json \
#        --config \
#        config/train/lalonde_cps/sample1/lalonde_cps.json \
#        --mask \
#        True \
#        --data_train_file \
#        data/realcause_datasets/lalonde_cps/sample${i}/train/lalonde_cps_train.csv \
#        --data_holdout_file \
#        data/realcause_datasets/lalonde_cps/sample${i}/holdout/lalonde_cps_holdout.csv \
#        --model_train_file \
#        experiments/model/lalonde_cps/sample${i}/model_cps_train.pth \
#        --model_holdout_file \
#        experiments/model/lalonde_cps/sample${i}/model_cps_holdout.pth
#done
#for i in {85..99}
#do
#python3 src/train.py \
#        --dag \
#        config/dag/lalonde_cps_dag.json \
#        --config \
#        config/train/lalonde_cps/sample${i}/lalonde_cps.json \
#        --mask \
#        False \
#        --data_train_file \
#        data/realcause_datasets/lalonde_cps/sample${i}/train/lalonde_cps_train.csv \
#       --data_holdout_file \
#        data/realcause_datasets/lalonde_cps/sample${i}/holdout/lalonde_cps_holdout.csv \
#        --model_train_file \
#        experiments/model/lalonde_cps/sample${i}/model_cps_train_nomask.pth \
#      --model_holdout_file \
#        experiments/model/lalonde_cps/sample${i}/model_cps_holdout_nomask.pth
#done
#for i in {0..10}
#do
#python3 src/predict.py \
#        --dag \
#        config/dag/lalonde_cps_dag.json \
#        --config \
#        config/train/lalonde_cps/sample1/lalonde_cps.json \
#        --mask \
#        True \
#       --data_train_file \
#        data/realcause_datasets/lalonde_cps/sample${i}/train/lalonde_cps_train.csv \
#        --data_holdout_file \
#        data/realcause_datasets/lalonde_cps/sample${i}/holdout/lalonde_cps_holdout.csv \
#        --model_train_file \
#        experiments/model/lalonde_cps/sample${i}/model_cps_train.pth \
#        --model_holdout_file \
#        experiments/model/lalonde_cps/sample${i}/model_cps_holdout.pth \
#        --output_file \
#        experiments/predict/lalonde_cps/sample${i}/predictions_cps.csv
#done
#for i in {0..99}
#do
#python3 src/predict.py \
#        --dag \
#        config/dag/lalonde_cps_dag.json \
#        --config \
#        config/train/lalonde_cps/sample${i}/lalonde_cps.json \
#        --mask \
#        False \
#        --data_train_file \
#        data/realcause_datasets/lalonde_cps/sample${i}/train/lalonde_cps_train.csv \
#        --data_holdout_file \
#        data/realcause_datasets/lalonde_cps/sample${i}/holdout/lalonde_cps_holdout.csv \
#        --model_train_file \
#        experiments/model/lalonde_cps/sample${i}/model_cps_train_nomask.pth \
#        --model_holdout_file \
#        experiments/model/lalonde_cps/sample${i}/model_cps_holdout_nomask.pth \
#        --output_file \
#        experiments/predict/lalonde_cps/sample${i}/predictions_cps_nomask.csv
#done
#for i in {0..10}
#do
#python3 src/evaluate.py \
#        --dag \
#        config/dag/lalonde_cps_dag.json \
#        --config \
#        config/train/lalonde_cps/sample1/lalonde_cps.json \
#        --data_file \
#        data/realcause_datasets/lalonde_cps/sample${i}/lalonde_cps_sample${i}.csv \
#        --predictions_file \
#        experiments/predict/lalonde_cps/sample${i}/predictions_cps.csv \
#        --results \
#       experiments/results/lalonde_cps/sample${i}/lalonde_cps.json
#done

#for i in {0..99}
#do
#python3 src/evaluate.py \
#        --dag \
#        config/dag/lalonde_cps_dag.json \
#        --config \
#        config/train/lalonde_cps/sample${i}/lalonde_cps.json \
#        --data_file \
#        data/realcause_datasets/lalonde_cps/sample${i}/lalonde_cps_sample${i}.csv \
#        --predictions_file \
#        experiments/predict/lalonde_cps/sample${i}/predictions_cps_nomask.csv \
#       --results \
#       experiments/results/lalonde_cps/sample${i}/lalonde_cps_nomask.json
#done
#for i in {0..99}
#do
#   rm "experiments/results/lalonde_cps/sample$i/lalonde_cps_nomask.json"
#done

#python3 src/evaluate_final.py \
#        --data_name \
#        lalonde_cps \
#        --results \
#        experiments/results/lalonde_cps/lalonde_cps.json

#python3 src/visualization/visualize.py \
#        --data_name \
#        lalonde_cps