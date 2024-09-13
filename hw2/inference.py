# Evaluation script for the summarization task.

import argparse
import jsonlines
import logging
import os

import datasets
import numpy as np
import torch
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    set_seed,
)


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on a summarization dataset.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="A file containing the validation data.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file of the prediction results.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=256,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=64,
        help="The maximum total sequence length for target text after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="Whether to pad all samples to model maximum sentence length. "
        "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
        "efficient on GPU but very bad for TPU.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    set_seed(42)

    data_files = {}
    if args.test_file is not None:
        data_files["test"] = args.test_file
    extension = "json"
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
    )

    # Load fine-tuned model and tokenizer
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        use_fast=not args.use_slow_tokenizer,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < args.max_source_length
    ):
        if args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {args.max_source_length}."
            )
            model.resize_position_embeddings(args.max_source_length)
        elif args.resize_position_embeddings:
            model.resize_position_embeddings(args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Get the column names for input/target.
    column_names = raw_datasets["test"].column_names
    text_column = args.text_column
    if text_column not in column_names:
        raise ValueError(f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}")

    padding = "max_length" if args.pad_to_max_length else False

    test_dataset = raw_datasets["test"]

    def preprocess_function(examples):
        # remove pairs where at least one record is None
        inputs = []
        for i in range(len(examples[text_column])):
            if examples[text_column][i]:
                inputs.append(examples[text_column][i])

        inputs = [prefix + input for input in inputs]
        model_inputs = tokenizer(
            inputs,
            max_length=args.max_source_length,
            padding=padding,
            truncation=True,
        )

        return model_inputs

    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on test dataset",
    )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Infer on the dataset
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with jsonlines.open(args.output_file, "w") as writer:
        for idx, sample in enumerate(test_dataset):
            model_input = torch.tensor([sample["input_ids"]])
            model_input = model_input.to(device)
            pred = model.generate(
                model_input,
                max_new_tokens=args.max_target_length,
                num_beams=5,
                do_sample=False,
            )
            pred = pred.cpu().numpy()
            pred = np.where(pred != -100, pred, tokenizer.pad_token_id)
            decoded_pred = tokenizer.decode(pred[0], skip_special_tokens=True).strip()
            # Write predictions to file
            writer.write({"title": decoded_pred, "id": raw_datasets["test"][idx]["id"]})
            logger.info(f"Step: {idx} / {len(test_dataset)} --- {int(idx / len(test_dataset) * 100)}%")


if __name__ == "__main__":
    main()
