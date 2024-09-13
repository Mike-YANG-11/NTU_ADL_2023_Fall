# Fine-Tune & Inference Instruction
## File Structure
- **README.md**
- **download.sh** :
Downloading the data and models.
- **run.sh** :
Run inference on the test data using the instruction-tuned model in this homework.
- **data/** :
Contains datasets.
- **adapter_checkpoint/** : 
This directory contains instruction-tuned adapter model.
- **qlora_fine_tune.py** : 
The script for fine-tuning the mt5-small model on summarization task.
- **inference.py** : 
The script for performing inference using the trained model.
- **inference_zero_shot.py** :
The script for performing inference using the zero-shot method.
- **inference_few_shot.py** :
The script for performing inference using the few-shot method.
- **plot_curve.py** :
The script for plotting the training curve of perplexity.
- **requirements.txt** :
The required packages for running the scripts.

```
homework_3_directory/
│
├── README.md
│
├── download.sh
│
├── run.sh
│
├── data/
|   ├── public_test.json
│   ├── private_test.json
│   ├── train.json
│
├── adapter_checkpoint/
│   ├── adapter_config.json/
│   ├── adapter_model.bin/
│   
├── qlora_fine_tune.py
│
├── inference.py
│
├── inference_zero_shot.py
│
├── inference_few_shot.py
│
├── plot_curve.py
│
├── requirements.txt
```

## Setup Environment
### To setup and activate the environment, we need to run the script below:
```
conda env create -f environment.yml
conda activate r12945039_hw3_env
```

## Instruction-Tuning Adapter Model
### To run the instruction-tuning adapter model, we need to run the script below:
```
python qlora_fine_tune.py \
    --model_name_or_path {pre_trained_llm_model_dir} \
    --output_dir {path_to_save_lora_model} \
    --dataset {path_to_train_file} \
    --dataset_format {dataset_format} \
    --lora_r={lora_rank} \
    --lora_alpha={lora_alpha} \ 
    --lora_dropout={lora_dropout_rate} \
    --per_device_train_batch_size={per_device_train_batch_size} \
    --gradient_accumulation_steps={gradient_accumulation_steps} \
    --max_steps={max_training_steps} \
    --learning_rate={learning_rate} \
    --lr_scheduler_type {lr_scheduler_type} \
    --report_to {reporting_package} \
    --logging_steps={logging_steps} \
    --save_steps={save_steps} \
    --save_total_limit={save_total_limit} \
    --num_beams={num_beams} \
```
For example, if we want to run instruction-tuning on the Taiwan-LLM using the dataset, run the script below:
```
python ./qlora_fine_tune.py \
    --model_name_or_path ./Taiwan-LLM-7B-v2.0-chat \
    --output_dir ./new_adapter_checkpoint \
    --dataset ./data/train.json \
    --dataset_format Taiwan_LLM_chat \
    --lora_r=4 \
    --lora_alpha=16 \
    --lora_dropout=0.05 \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=16 \
    --max_steps=200 \
    --learning_rate=2e-4 \
    --lr_scheduler_type linear \
    --report_to tensorboard \
    --logging_steps=5 \
    --save_steps=10 \
    --save_total_limit=50 \
    --num_beams=3 \
```
The output adapter would be saved at "./new_adapter_checkpoint" folder.
The training logs file would be saved at "./new_adapter_checkpoint/runs" folder, which could be visualized using command:
```
tensorboard --logdir ./new_adapter_checkpoint/runs
```

## Inferring Test Data Using the Trained Models
### To run inference on test data, we need to run the script below:
```
python ./inference.py \
    --base_model_path {pre_trained_llm_model_dir} \
    --peft_path {path_to_peft_model} \
    --test_data_path {path_to_test_file} \
    --output_path {path_to_save_prediction_file} \

```
For example, if we want to run inference on the ./data/public_test.json file using the instruction-tuned model, run the script below:
```
python ./inference.py \
    --base_model_path ./Taiwan-LLM-7B-v2.0-chat \
    --peft_path ./adapter_checkpoint \
    --test_data_path ./data/public_test.json \
    --output_path ./results/prediction.json \
```
The prediction output json file would be saved at the "./results" folder.

## Inferring Test Data Using the Zero-Shot Method
### To run zero-shot inference on first 10th test data, we need to run the script below:
```
python ./inference_zero_shot.py \
    --base_model_path {pre_trained_llm_model_dir} \
    --test_data_path {path_to_test_file} \
    --output_path {path_to_save_prediction_file} \

```
For example, if we want to run inference on the first 10 data in ./data/public_test.json file using the zero-shot method, run the script below:
```
python ./inference_zero_shot.py \
    --base_model_path ./Taiwan-LLM-7B-v2.0-chat \
    --test_data_path ./data/public_test.json \
    --output_path ./results/prediction_zero_shot.json \
```
The prediction output json file would be saved at the "./results" folder.

## Inferring Test Data Using the Few-Shot Method
### To run few-shot inference on first 10th test data, we need to run the script below:
```
python ./inference_few_shot.py \
    --base_model_path {pre_trained_llm_model_dir} \
    --test_data_path {path_to_test_file} \
    --output_path {path_to_save_prediction_file} \

```
For example, if we want to run inference on the first 10 data in ./data/public_test.json file using the few-shot method, run the script below:
```
python ./inference_few_shot.py \
    --base_model_path ./Taiwan-LLM-7B-v2.0-chat \
    --test_data_path ./data/public_test.json \
    --output_path ./results/prediction_few_shot.json \
```
The prediction output json file would be saved at the "./results" folder.

## Plotting the Training Curve of Perplexity
### To plot the training curve of perplexity on public test set, we need to run the script below:
```
python ./plot_curve.py \
    --base_model_path {pre_trained_llm_model_dir} \
    --peft_path {path_to_peft_model_checkpoints} \
    --test_data_path {path_to_test_file} \

```
For example, if we want to plot the training curve of perplexity on ./data/public_test.json using the instruction-tuned model we've just showed above, run the script below:
```
python ./plot_curve.py \
    --base_model_path ./Taiwan-LLM-7B-v2.0-chat \
    --peft_path ./new_adapter_checkpoint \
    --test_data_path ./data/public_test.json \
```
The training curve of perplexity would be saved in the "./ppl_learning_curve.png" image.