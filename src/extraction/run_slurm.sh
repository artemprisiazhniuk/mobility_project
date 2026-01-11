#!/bin/bash
#SBATCH --job-name=roberta_got_container_job
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=output_logs/output_%j.log
#SBATCH --error=error_logs/error_%j.log
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gres=gpu:1

echo $SLURM_SUBMIT_HOST


CONTAINER_IMAGE="./mobility_oneclass.sif"
RUN_TYPE="train" # train, eval, inference


if [ "$RUN_TYPE" == "train" ]; then
  echo "Training mode selected."
  
  MODEL_NAME="FacebookAI/xlm-roberta-large" # -finetuned-conll03-german" # german
  # MODEL_NAME="FacebookAI/roberta-large"
  # MODEL_NAME="google-bert/bert-large-cased"
  MODEL_NAME_SHORT=$(basename "$MODEL_NAME")
else
  echo "Evaluation/inference mode selected."
  MODEL_NAME="../../models/extraction/xlm-roberta-large/xlm-roberta-large-r4a16d0_3/best_model/"
  MODEL_NAME_SHORT=$(basename "$(dirname "$MODEL_NAME")")
fi

RATIO="4"
DATA_TYPE="r-MANUAL_FV_mixed_$RATIO"
DATA_PATH="../../data/extraction/MANUAL_FV_mixed_$RATIO"

echo $DATA_TYPE

LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
LORA_DROPOUT_STR="${LORA_DROPOUT/./_}"
RUN_NAME="$MODEL_NAME_SHORT-$DATA_TYPE-r${LORA_R}a${LORA_ALPHA}d${LORA_DROPOUT_STR}"
EXPERIMENT_NAME="Mobility-new"
MLRUNS_PATH="../../mlruns"
SYNONYMS_FILE="../../data/synonyms.json"

echo $EXPERIMENT_NAME

if [ "$RUN_TYPE" == "train" ]; then
  SAVE_PATH_PREFIX="../../models/extraction"
else
  SAVE_PATH_PREFIX="../../data/inference"
fi
SAVE_PATH="$SAVE_PATH_PREFIX/$MODEL_NAME_SHORT/$RUN_NAME"

mkdir -p "$SAVE_PATH"

if [ "$RUN_TYPE" == "train" ]; then
  echo "Running training..."

  apptainer exec --nv \
  --writable-tmpfs \
  --bind "$DATA_PATH:/data/input_data" \
  --bind "$SAVE_PATH:/model" \
  --bind "$MLRUNS_PATH:/mlruns" \
  --bind "$SYNONYMS_FILE:/data/synonyms.json" \
  "$CONTAINER_IMAGE" \
  bash -c "python3 /app/train.py \
  --model-name $MODEL_NAME \
  --data-path /data/input_data \
  --save-path /model \
  --batch-size 16 \
  --num-epochs 100 \
  --learning-rate 2e-4 \
  --warmup-ratio 0.05 \
  --weight-decay 0.01 \
  --lr-scheduler-type cosine \
  --logging-steps 200 \
  --eval-steps 1000 \
  --use-lora \
  --lora-r $LORA_R \
  --lora-alpha $LORA_ALPHA \
  --lora-dropout $LORA_DROPOUT \
  --train-classifier \
  --add-prefix-space \
  --synonyms-file /data/synonyms.json \
  --use-fast-tokenizer \
  --use-mlflow \
  --mlruns-path /mlruns \
  --mlflow-run-name $RUN_NAME \
  --mlflow-experiment-name $EXPERIMENT_NAME \
  --entity2contraction '{\"roadways\":\"ROAD\"}'
  "
  # --entity2contraction '{\"roadways\":\"ROAD\"}'

elif [ "$RUN_TYPE" == "eval" ]; then
  echo "Running evaluation..."

  apptainer exec --nv \
  --writable-tmpfs \
  --bind "$DATA_PATH:/data/input_data" \
  --bind "$SAVE_PATH:/labels" \
  "$CONTAINER_IMAGE" \
  bash -c "python3 /app/inference.py \
  --run-type eval \
  --model-name $MODEL_NAME\
  --save-path /labels \
  --bf16 \
  --use-fast-tokenizer \
  --data-path /data/input_data \
  --label-path /data/input_labels \
  --batch-size 64 \
  --eval-batch-size 64 \
  --eval-empty \
  --eval-full
  "
  # --entity2contraction '{"roadways": "ROAD"}'
elif [ "$RUN_TYPE" == "inference" ]; then
  echo "Running inference..."

  apptainer exec --nv \
  --writable-tmpfs \
  --bind "$DATA_PATH:/data/input_data" \
  --bind "$SAVE_PATH:/labels" \
  "$CONTAINER_IMAGE" \
  bash -c "python3 /app/inference.py \
  --run-type inference \
  --model-name $MODEL_NAME\
  --save-path /labels \
  --use-fast-tokenizer \
  --data-path /data/input_data \
  --batch-size 64 \
  --eval-batch-size 64
  "
  # --eval-full
else
  echo "Invalid run type specified. Use 'train', 'eval', or 'inference'."
  exit 1
fi
