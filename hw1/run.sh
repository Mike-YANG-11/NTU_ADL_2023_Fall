#!/bin/bash
# This script is used to run inference on the test set.
python ./inference.py \
    --ps_model_name_or_path \
    ./models/fine_tuned_models/paragraph_selection/chinese_macbert_base \
    --ss_model_name_or_path \
    ./models/fine_tuned_models/span_selection/chinese_macbert_base \
    --context_file \
    $1 \
    --test_file \
    $2 \
    --output_file \
    $3 \
    --pad_to_max_length \
    --max_seq_length \
    512 \
    --max_answer_length \
    40