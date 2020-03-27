#!/bin/bash
pwd

NUM_EPOCHS=5
NUM_TURNS=5

MODEL_PATH="kbert"
mkdir -p ../models/${MODEL_PATH}
python gpt_tuning.py --output_dir ${MODEL_PATH} --num_train_epochs ${NUM_EPOCHS} --num_turns ${NUM_TURNS} --kbert
python gpt_sample.py --model_dir ${MODEL_PATH} --output_dir ${MODEL_PATH} --num_turns ${NUM_TURNS} --top_p 0.95 --kbert

MODEL_PATH="kbert_position"
mkdir -p ../models/${MODEL_PATH}
python gpt_tuning.py --output_dir ${MODEL_PATH} --num_train_epochs ${NUM_EPOCHS} --num_turns ${NUM_TURNS} --kbert --kbert_position
python gpt_sample.py --model_dir ${MODEL_PATH} --output_dir ${MODEL_PATH} --num_turns ${NUM_TURNS} --top_p 0.95 --kbert --kbert_position

MODEL_PATH="kbert_mask_position"
mkdir -p ../models/${MODEL_PATH}
python gpt_tuning.py --output_dir ${MODEL_PATH} --num_train_epochs ${NUM_EPOCHS} --num_turns ${NUM_TURNS} --kbert --kbert_position --kbert_mask
python gpt_sample.py --model_dir ${MODEL_PATH} --output_dir ${MODEL_PATH} --num_turns ${NUM_TURNS} --top_p 0.95 --kbert --kbert_position --kbert_mask

#echo "Finished."

