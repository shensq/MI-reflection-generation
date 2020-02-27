#!/bin/bash
pwd

# python retrieve_candidate.py --model_dir mi_nli

mkdir -p ../models/mi_tuned_5turn
python gpt_tuning.py --output_dir mi_tuned_5turn --num_train_epochs 10 --num_turns 5
python gpt_sample.py --model_dir mi_tuned_5turn --output_dir mi_tuned_5turn --num_turns 5 --top_p 0.95

mkdir -p ../models/mi_tuned_aug
python gpt_tuning.py --output_dir mi_tuned_aug --num_train_epochs 10 --num_turns 5 --augment
python gpt_sample.py --model_dir mi_tuned_aug --output_dir mi_tuned_aug --num_turns 5 --augment  --top_p 0.95

# mkdir -p ../models/mi_tuned_keyword
#python gpt_tuning.py --output_dir mi_tuned_keyword --num_train_epochs 10 --num_turns 5 --keyword
# python gpt_sample.py --model_dir mi_tuned_keyword --output_dir mi_tuned_keyword --num_turns 5 --keyword  --top_p 0.95

# mkdir -p ../models/mi_tuned_both
# python gpt_tuning.py --output_dir mi_tuned_both --num_train_epochs 10 --num_turns 10 --keyword --augment
# python gpt_sample.py --model_dir mi_tuned_both --output_dir mi_tuned_both --num_turns 10 --keyword --augment --top_p 0.95
echo "Finished."
