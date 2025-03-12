"""
Preprocess Social IQa dataset for Logic-RL framework.
"""

import os
import json
import argparse
from datasets import Dataset, load_dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs


def make_prefix(dp, template_type):
    """
    Generate prompt prefix for Social IQa dataset.
    
    Args:
        dp: A single data point (dictionary) from the Social IQa dataset.
        template_type: Type of template to use ('base' or 'qwen-instruct' or 'llama-instruct').
    
    Returns:
        str: Generated prefix string.
    """
    context = dp['context']
    question = dp['question']
    answers = dp['answers']

    if template_type == 'base': 
        prefix = (
            f"""The user asks a context-dependent, multiple-choice question. """
            f"""The assistant first reasons through the question and then selects the final answer. """
            f"""The reasoning process is enclosed within <think>...</think> tags, and the answer is enclosed within <answer>...</answer> tags. """
            f"""For example: \"<think> reasoning process here </think> <answer> (D) </answer>\". """
            f"""The user now asks the assistant to reason through and answer the following problem:\n"""
            f"""User: Based on the context: \"{context}\", answer the question: \"{question}\". """
            f"""Choose one of the following options: (A) {answers[0]}, (B) {answers[1]}, (C) {answers[2]}. """
            f"""Express your reasoning process and provide the final answer.\n"""
            f"""Assistant: <think>"""
        )
    elif template_type == 'qwen-instruct':
        prefix = (
            f"""<|im_start|>system\nYou are a helpful assistant. """
            f"""The user asks a context-dependent, multiple-choice question. """
            f"""The assistant first reasons through the question and then selects the final answer. """
            f"""The reasoning process is enclosed within <think>...</think> tags, and the answer is enclosed within <answer>...</answer> tags. """
            f"""For example: \"<think> reasoning process here </think> <answer> (D) </answer>.\" """
            f"""The user now asks the assistant to reason through and answer the following problem:\n<|im_end|>\n"""
            f"""<|im_start|>user\nBased on the context: \"{context}\", answer the question: \"{question}\". Choose one of the following options: (A) {answers[0]}, (B) {answers[1]}, (C) {answers[2]}. Express your reasoning process and provide the final answer.\n<|im_end|>\n"""
            f"""<|im_start|>assistant\n<think>"""
        )
    elif template_type == 'llama-instruct':
        prefix = (
            f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant. """
            f"""The user asks a context-dependent, multiple-choice question. """
            f"""The assistant first reasons through the question and then selects the final answer. """
            f"""The reasoning process is enclosed within <think>...</think> tags, and the answer is enclosed within <answer>...</answer> tags. """
            f"""For example: \"<think> reasoning process here </think> <answer> (D) </answer>.\" """
            f"""The user now asks the assistant to reason through and answer the following problem:\n"""
            f"""<|start_header_id|>user<|end_header_id|>\nBased on the context: \"{context}\", answer the question: \"{question}\". Choose one of the following options: (A) {answers[0]}, (B) {answers[1]}, (C) {answers[2]}. Express your reasoning process and provide the final answer.\n"""
            f"""<|start_header_id|>assistant<|end_header_id|>\n<think>"""
        )
    return prefix


def download_social_iqa(data_path):
    """
    Download the Social IQa dataset using Hugging Face's `datasets` library if not already present.

    Args:
        data_path (str): Path to the directory where the raw dataset should be saved.
    """
    os.makedirs(data_path, exist_ok=True)
    train_file = os.path.join(data_path, "train.jsonl")
    test_file = os.path.join(data_path, "test.jsonl")

    # Check if files already exist
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("Downloading Social IQa dataset...")
        dataset = load_dataset("allenai/social_i_qa")
        dataset["train"].to_json(train_file, orient="records", lines=True)
        dataset["validation"].to_json(test_file, orient="records", lines=True)
        print(f"Social IQa dataset downloaded and saved to {data_path}")
    else:
        print(f"Dataset already exists in {data_path}")


def preprocess_social_iqa(input_dir, output_dir, split, template_type):
    """
    Preprocess Social IQa dataset for Logic-RL.
    
    Args:
        input_dir (str): Path to the raw Social IQa dataset directory.
        output_dir (str): Path to save the processed dataset.
        split (str): Dataset split ('train' or 'test').
        template_type (str): The type of template to use ('base' or 'qwen-instruct' or 'llama-instruct').
    """
    input_file = os.path.join(input_dir, f"{split}.jsonl")
    output_file = os.path.join(output_dir, f"{split}.parquet")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process dataset
    processed_data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in tqdm(enumerate(f), desc=f"Processing {split} split"):
            data = json.loads(line)
            processed_data.append({
                "data_source": "social_iqa",
                "prompt": [{
                    "role": "user",
                    "content": make_prefix({
                        "context": data["context"],
                        "question": data["question"],
                        "answers": [data["answerA"], data["answerB"], data["answerC"]]
                    }, template_type=template_type)
                }],
                "ability": "reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {  # Change ground_truth to a dictionary
                        "label": data["label"],
                        "answerA": data["answerA"],
                        "answerB": data["answerB"],
                        "answerC": data["answerC"]
                    }
                },
                "extra_info": {
                    "split": split,
                    "index": idx
                }
            })

    # Save processed data
    dataset = Dataset.from_list(processed_data)
    dataset.to_parquet(output_file)
    print(f"Processed {split} data saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess Social IQa dataset for Logic-RL.")
    parser.add_argument('--local_dir', type=str, required=True, help="Path to save processed data.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to raw Social IQa data.")
    parser.add_argument('--template_type', type=str, default='base', help="Template type ('base' or 'qwen-instruct' or 'llama-instruct').")
    parser.add_argument('--hdfs_dir', type=str, default=None, help="Path to HDFS directory (if using HDFS).")
    args = parser.parse_args()

    # Download dataset if not already available
    download_social_iqa(args.data_path)

    # Process train and test splits
    for split in ["train", "test"]:
        preprocess_social_iqa(args.data_path, args.local_dir, split, args.template_type)

    # Optionally copy to HDFS
    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)