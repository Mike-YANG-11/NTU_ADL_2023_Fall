#!/bin/bash
# This script is used to run inference on the test set.
python ./inference.py \
    --model_name_or_path ./models/mt5-small-summarization \
    --test_file $1 \
    --output_file $2 \
    --text_column maintext \
    --source_prefix "summarize: " \
    --pad_to_max_length 