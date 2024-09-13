import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from peft import PeftModel
from utils import get_prompt, get_bnb_config
import argparse
import matplotlib.pyplot as plt


def perplexity(
    model,
    tokenizer,
    data,
    max_length=2048,
):
    data_size = len(data)
    instructions = [get_prompt(x["instruction"]) for x in data]
    outputs = [x["output"] for x in data]

    # Tokenize data
    tokenized_instructions = tokenizer(instructions, add_special_tokens=False)
    tokenized_outputs = tokenizer(outputs, add_special_tokens=False)
    output_masks = []

    # Format data
    for i in range(data_size):
        instruction_input_ids = [tokenizer.bos_token_id] + tokenized_instructions["input_ids"][i]
        output_input_ids = tokenized_outputs["input_ids"][i] + [tokenizer.eos_token_id]
        tokenized_instructions["input_ids"][i] = instruction_input_ids + output_input_ids
        tokenized_instructions["attention_mask"][i] = [1] * len(tokenized_instructions["input_ids"][i])
        output_mask = [0] * len(instruction_input_ids) + [1] * len(output_input_ids)

        tokenized_instructions["input_ids"][i] = torch.tensor(tokenized_instructions["input_ids"][i][:max_length])
        tokenized_instructions["attention_mask"][i] = torch.tensor(
            tokenized_instructions["attention_mask"][i][:max_length]
        )
        output_mask = torch.tensor(output_mask[:max_length])
        output_masks.append(output_mask)

    # Calculate ppl
    ppls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for i in tqdm(range(data_size)):
        input_ids = tokenized_instructions["input_ids"][i].unsqueeze(0)
        attn_mask = tokenized_instructions["attention_mask"][i].unsqueeze(0)
        output_mask = output_masks[i].unsqueeze(0)
        label = input_ids

        with torch.no_grad():
            out_logits = model(input_ids, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_label = label[..., 1:].contiguous()
        shift_output_mask = output_mask[..., 1:].contiguous()
        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_label) * shift_output_mask).sum(1) / shift_output_mask.sum(1)
        )
        ppls += perplexity_batch.tolist()
    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9).",
    )
    parser.add_argument("--peft_path", type=str, required=True, help="Path to the saved PEFT checkpoints.")
    parser.add_argument("--test_data_path", type=str, default="", required=True, help="Path to test data.")
    args = parser.parse_args()

    checkpoints = []
    for subdir in os.listdir(args.peft_path):
        if subdir[:10] == ("checkpoint"):
            checkpoints.append(subdir)
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

    bnb_config = get_bnb_config()
    ppl_list = []
    for checkpoint in checkpoints:
        # Load model
        if args.base_model_path:
            base_model = AutoModelForCausalLM.from_pretrained(
                args.base_model_path, torch_dtype=torch.bfloat16, quantization_config=bnb_config
            )
            tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
        else:
            model_name = "yentinglin/Taiwan-LLM-7B-v2.0-chat"
            revision = "5073b2bbc1aa5519acdc865e99832857ef47f7c9"
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name, revision=revision, torch_dtype=torch.bfloat16, quantization_config=bnb_config
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                revision=revision,
            )

        # Load LoRA model
        model = PeftModel.from_pretrained(base_model, os.path.join(args.peft_path, checkpoint))
        print(f"Load checkpoint: {checkpoint}")

        with open(args.test_data_path, "r") as f:
            data = json.load(f)

        model.eval()
        ppl = perplexity(model, tokenizer, data)
        ppl_list.append(ppl["mean_perplexity"])

        del base_model, tokenizer, model

    # Plot ppl curve
    steps = np.arange(
        int(checkpoints[0].split("-")[1]),
        int(checkpoints[-1].split("-")[1]) + int(checkpoints[0].split("-")[1]),
        int(checkpoints[0].split("-")[1]),
    )
    print(steps)
    plt.plot(steps, ppl_list)
    plt.title("Learning Curve on the Public Testing Set")
    plt.xlabel("Steps")
    plt.ylabel("Perplexity")
    plt.savefig("ppl_learning_curve.png")
    plt.show()
