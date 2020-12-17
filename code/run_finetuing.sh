#!/bin/bash
pwd

NUM_EPOCHS=5
NUM_TURNS=5

MODEL_PATH="test_experiment"${NUM_EPOCHS}
mkdir -p ../experiments/${MODEL_PATH}
python gpt_tuning.py --output_dir ${MODEL_PATH} --num_train_epochs ${NUM_EPOCHS} --num_turns ${NUM_TURNS}
python gpt_sample.py --model_dir ${MODEL_PATH} --output_dir ${MODEL_PATH} --num_turns ${NUM_TURNS} --top_p 0.95

echo "Finished."

