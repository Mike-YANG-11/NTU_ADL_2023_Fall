import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import json
from utils import get_prompt_zero_shot, get_bnb_config
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9).",
    )
    parser.add_argument("--test_data_path", type=str, default="", required=True, help="Path to test data.")
    parser.add_argument("--output_path", type=str, default="", required=True, help="Path to prediction result.")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Load model
    bnb_config = get_bnb_config()
    if args.base_model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path, torch_dtype=torch.bfloat16, quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    else:
        model_name = "yentinglin/Taiwan-LLM-7B-v2.0-chat"
        revision = "5073b2bbc1aa5519acdc865e99832857ef47f7c9"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, torch_dtype=torch.bfloat16, quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    set_seed(42)

    # Load test data
    with open(args.test_data_path, "r") as f:
        data = json.load(f)

    # Run inferece
    model.eval()
    with open(args.output_path, "w", encoding="utf-8") as output_file:
        output_list = []
        for i in tqdm(range(10)):
            prompt = get_prompt_zero_shot(data[i]["instruction"])
            inputs = tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                outputs = model.generate(**inputs)
            text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            text = text.split("專業古文學者:")
            output_list.append({"id": data[i]["id"], "output": text[-1]})

        # Write output to file
        json.dump(output_list, output_file, ensure_ascii=False)
