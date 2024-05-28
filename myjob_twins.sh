#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --output=experiments/results/output_%j.txt
#SBATCH --error=experiments/results/error_%j.txt
#SBATCH -c 15
#SBATCH -t 24:00:00
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
#pip3 install doubleml
#python3 src/doubly_robust_baseline.py
#for i in {85..99}
#do
#  python3 src/doubly_robust_baseline.py \
#          --data_name \
#          twins \
#          --data_file \
#          data/realcause_datasets/twins/sample${i}/twins_sample${i}.csv \
#          --results \
#          experiments/results/twins/sample${i}/twins.json
#done
#for i in {0..99}
#do
#python3 src/dataset.py \
#        --dag \
#        config/dag/twins_dag.json \
#        --data_file \
#        data/realcause_datasets/twins/sample${i}/twins_sample${i}.csv \
#        --train_output_file \
#        data/realcause_datasets/twins/sample${i}/train/twins_train.csv \
#        --holdout_output_file \
#        data/realcause_datasets/twins/sample${i}/holdout/twins_holdout.csv
#done
for i in {0..99}
do
  python3 experiments/tuning/fine_tune_twins.py \
          --dag \
          config/dag/twins_dag.json \
          --data_train_file \
          data/realcause_datasets/twins/sample${i}/train/twins_train.csv \
          --data_holdout_file \
          data/realcause_datasets/twins/sample${i}/holdout/twins_holdout.csv \
          --results \
          config/train/twins/sample${i}/twins.json
done
#for i in {0..99}
#do
#  python3 src/train.py \
#          --dag \
#          config/dag/twins_dag.json \
#          --config \
#          config/train/twins/sample${i}/twins.json \
#          --mask \
#          True \
#          --data_train_file \
#          data/realcause_datasets/twins/sample${i}/train/twins_train.csv \
#          --data_holdout_file \
#          data/realcause_datasets/twins/sample${i}/holdout/twins_holdout.csv \
#          --model_train_file \
#          experiments/model/twins/sample${i}/model_twins_train.pth \
#          --model_holdout_file \
#          experiments/model/twins/sample${i}/model_twins_holdout.pth
#done
#for i in {0..99}
#do
#  python3 src/train.py \
#          --dag \
#          config/dag/twins_dag.json \
#          --config \
#          config/train/twins/sample${i}/twins.json \
#          --mask \
#          False \
#          --data_train_file \
#          data/realcause_datasets/twins/sample${i}/train/twins_train.csv \
#          --data_holdout_file \
#          data/realcause_datasets/twins/sample${i}/holdout/twins_holdout.csv \
#          --model_train_file \
#          experiments/model/twins/sample${i}/model_twins_train_nomask.pth \
#          --model_holdout_file \
#          experiments/model/twins/sample${i}/model_twins_holdout_nomask.pth
#done
#done
#for i in {0..99}
#do
#python3 src/predict.py \
#        --dag \
#        config/dag/twins_dag.json \
#        --config \
#       config/train/twins/sample${i}/twins.json \
#        --mask \
#        False \
#        --data_train_file \
#       data/realcause_datasets/twins/sample${i}/train/twins_train.csv \
#        --data_holdout_file \
#        data/realcause_datasets/twins/sample${i}/holdout/twins_holdout.csv \
#        --model_train_file \
#       experiments/model/twins/sample${i}/model_twins_train_nomask.pth \
#       --model_holdout_file \
#        experiments/model/twins/sample${i}/model_twins_holdout_nomask.pth \
#        --output_file \
#       experiments/predict/twins/sample${i}/predictions_twins_nomask.csv
#done
#for i in {0..99}
#do
#python3 src/evaluate.py \
#        --dag \
#        config/dag/twins_dag.json \
#        --config \
#        config/train/twins/sample${i}/twins.json \
#        --data_file \
#        data/realcause_datasets/twins/sample${i}/twins_sample${i}.csv \
#        --predictions_file \
#        experiments/predict/twins/sample${i}/predictions_twins_nomask.csv \
#        --results \
#        experiments/results/twins/sample${i}/twins_nomask.json
#done
#for i in {0..99}
#do
#   mkdir "experiments/results/twins/sample$i"
#done
#python3 src/evaluate_final.py \
#        --data_name \
#        twins \
#        --results \
#        experiments/results/twins/twins.json
python3 src/visualization/visualize.py \
        --data_name \
        twins