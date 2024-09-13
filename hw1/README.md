# Fine-Tune & Inference Instruction
## File Structure
- **download.sh** :
Downloading the data and models.
- **run.sh** :
Run inference on the test data using the best fine-tuned model in this homework.
- **data/** :
Contains train, validation, and test data.
- **models/** : 
This directory contains pre-trained BERT models, fine-tuned BERT models, model configurations from scratch and models trained from scratch.
- **ps_fine_tune.py** : 
The script for fine-tuning the BERT model on paragraph selection task.
- **ss_fine_tune.py** : 
The script for fine-tuning the BERT model on span selection task.
- **inference_script.py** : 
The script for performing inference using the trained model.

```
homework_1_directory/
│
├── download.sh
│
├── run.sh
│
├── data/
|   ├── context.json
│   ├── train.json
│   ├── valid.json
│   ├── test.json
│
├── models/
│   ├── pretrained_models/
│   │   ├── hfl_chinese_macbert_base/
│   │   │   ├── config.json
│   │   │   ├── tokenizer.json
│   │   │   ├── pytorch.bin
│   │   │   ├── ...
│   │   ├── hfl_chinese_roberta_wwm_ext/
│   │       ├── config.json
│   │       ├── tokenizer.json
│   │       ├── pytorch.bin
│   │       ├── ...
│   │
│   ├── fine_tuned_models/
│   │   ├── paragraph_selection/
│   │   │   ├── chinese_macbert_base/
│   │   │   │   ├── config.json
│   │   │   │   ├── tokenizer.json
│   │   │   │   ├── pytorch.bin
│   │   │   │   ├── ...
│   │   │   ├── chinese_roberta_wwm_ext/
│   │   │       ├── config.json
│   │   │       ├── tokenizer.json
│   │   │       ├── pytorch.bin
│   │   │       ├── ...
│   │   │
│   │   ├── span_selection/
│   │       ├── chinese_macbert_base/
│   │       │   ├── config.json
│   │       │   ├── tokenizer.json
│   │       │   ├── pytorch.bin
│   │       │   ├── ...
│   │       ├── chinese_roberta_wwm_ext/
│   │           ├── config.json
│   │           ├── tokenizer.json
│   │           ├── pytorch.bin
│   │           ├── ...
│   │
│   ├── from_scrtch_config/
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   ├── ...
│   │
│   ├── from_scratch_models/
│       ├── paragraph_selection/
│           ├── config.json
│           ├── tokenizer.json
│           ├── pytorch.bin
│           ├── ...
│
├── ps_fine_tune.py
├──
├── ss_fine_tune.py
├──
├── inference.py
```


## Fine-Tuning
### To fine-tune the BERT model on "Paragraph Selection" task, we need to run the script below:
```
python ps_fine_tune.py \
    --model_name_or_path {saved_model_dir} \
    --context_file {path_to_context_file} \
    --train_file {path_to_train_file} \
    --validation_file {path_to_validation_file} \
    --output_dir {dir_to_save_fine_tuned_model_and_results} \
    --max_seq_length {sequence_max_length} \
    --pad_to_max_length \
    --per_device_train_batch_size {batch_size_per_device} \
    --gradient_accumulation_steps {minimum_steps_to_update_model} \
    --learning_rate {learning_rate} \
    --num_train_epochs {epochs}
```
For example, if we want to fine-tuned the "MacBERT pre-trained" model on "Paragraph Selection" task, run the script below:
```
python ps_fine_tune.py \
    --model_name_or_path ./models/pretrained_models/hfl_chinese_macbert_base \
    --context_file ./data/context.json \
    --train_file ./data/train.json \
    --validation_file ./data/valid.json \
    --output_dir ./new_models/fine_tuned_models/paragraph_selection/chinese_macbert_base \
    --max_seq_length 512 \
    --pad_to_max_length \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --learning_rate 3e-5 \
    --num_train_epochs 3
```
The output model configurations and weights would be saved at "./new_models/fine_tuned_models/paragraph_selection/chinese_macbert_base" folder.

##
### To fine-tune the BERT model on "Span Selection" task, we need to run the script below:
```
python ss_fine_tune.py \
    --model_name_or_path {saved_model_dir} \
    --context_file {path_to_context_file} \
    --train_file {path_to_train_file} \
    --validation_file {path_to_validation_file} \
    --output_dir {dir_to_save_fine_tuned_model_and_results} \
    --max_seq_length {sequence_max_length} \
    --pad_to_max_length \
    --max_answer_length {output_span_max_length} \
    --per_device_train_batch_size {batch_size_per_device} \
    --gradient_accumulation_steps {minimum_steps_to_update_model} \
    --learning_rate {learning_rate} \
    --num_train_epochs {epochs}
```
For example, if we want to fine-tuned the "MacBERT pre-trained" model on "Span Selection" task, run the script below:
```
python ss_fine_tune.py \
    --model_name_or_path ./models/pretrained_models/hfl_chinese_macbert_base \
    --context_file ./data/context.json \
    --train_file ./data/train.json \
    --validation_file ./data/valid.json \
    --output_dir ./new_models/fine_tuned_models/span_selection/chinese_macbert_base \
    --max_seq_length 512 \
    --pad_to_max_length \
    --max_answer_length 40 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --learning_rate 3e-5 \
    --num_train_epochs 3
```
The output models configuration and weights would be saved at "./new_models/fine_tuned_models/span_selection/chinese_macbert_base" folder.


## Inferring Test Data Using the Trained Models
### To run inference on test data, we need to run the script below:
```
python inference.py \
    --ps_model_name_or_path {fine_tuned_paragraph_selection_model_dir} \
    --ss_model_name_or_path {fine_tuned_span_selection_model_dir} \
    --context_file {path_to_context_file} \
    --test_file {path_to_test_file} \
    --output_file {prediction_output_file} \
    --max_seq_length {sequence_max_length} \
    --pad_to_max_length \
    --max_answer_length {output_span_max_length}

```
For example, if we want to run inference using the fine-tuned MacBERT model we've just showed above, run the script below:
```
python inference.py \
    --ps_model_name_or_path ./new_models/fine_tuned_models/paragraph_selection/chinese_macbert_base \
    --ss_model_name_or_path ./new_models/fine_tuned_models/span_selection/chinese_macbert_base \
    --context_file ./data/context.json \
    --test_file ./data/test.json \
    --output_file ./results/prediction.csv \
    --max_seq_length 512 \
    --pad_to_max_length \
    --max_answer_length 40
```
The prediction output CSV file would be saved at the "./results" folder.


## Train Models from Scrtch
### To train the BERT model on "Paragraph Selection" task from scrtch, we need to run the script below:
```
python ps_fine_tune.py \
    --config_name {model_config_dir} \
    --tokenizer_name {tokenizer_config_dir} \
    --context_file {path_to_context_file} \
    --train_file {path_to_train_file} \
    --validation_file {path_to_validation_file} \
    --output_dir {dir_to_save_fine_tuned_model_and_results} \
    --max_seq_length {sequence_max_length} \
    --pad_to_max_length \
    --per_device_train_batch_size {batch_size_per_device} \
    --gradient_accumulation_steps {minimum_steps_to_update_model} \
    --learning_rate {learning_rate} \
    --num_train_epochs {epochs}
```
For example, if we want to train a model on "Paragraph Selection" task from scratch, run the script below:
```
python ps_fine_tune.py \
    --config_name ./models/from_scratch_config \
    --tokenizer_name ./models/from_scratch_config \
    --context_file ./data/context.json \
    --train_file ./data/train.json \
    --validation_file ./data/valid.json \
    --output_dir ./new_models/from_scratch_models/paragraph_selection_model \
    --max_seq_length 512 \
    --pad_to_max_length \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --learning_rate 3e-5 \
    --num_train_epochs 3
```
The output model configurations and weights would be saved at "./new_models/from_scratch_models/paragraph_selection_model" folder.