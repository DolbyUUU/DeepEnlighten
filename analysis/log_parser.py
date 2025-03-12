import argparse
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Default output file
OUTPUT_FILE = "parsed_logs.json"

# Possible assistant headers
assistant_headers = [
    "Assistant:",
    "<|im_start|>assistant",
    "<|start_header_id|>assistant<|end_header_id|>"
]


def extract_solution(solution_str: str) -> Optional[str]:
    """
    Extract the reasoning (<think>) and final answer (<answer>) from the response.

    Args:
        solution_str: Raw response string from the language model.

    Returns:
        A tuple of (reasoning, final_answer) or None if extraction fails.
    """
    # Find the correct header
    matched_header = None
    start_index = -1
    for header in assistant_headers:
        if header in solution_str:
            start_index = solution_str.find(header)
            matched_header = header
            break

    if start_index != -1 and matched_header:
        # Extract the portion after the header
        solution_str = solution_str[start_index + len(matched_header):].strip()
    else:
        print("[Error] Failed to locate model response header. Check for unexpected formats.")
        return None

    # Ensure <think> and <answer> tags are present
    think_pattern = re.compile(r"<think>(.*?)</think>", re.S)
    answer_pattern = re.compile(r"<answer>\s*\((.*?)\)\s*(.*?)</answer>", re.S)

    think_match = think_pattern.search(solution_str)
    answer_match = answer_pattern.search(solution_str)

    if not think_match:
        print("[Warning] <think> tags not found. Skipping reasoning extraction.")
        reasoning = None
    else:
        reasoning = think_match.group(1).strip()

    if not answer_match:
        print("[Warning] <answer> tags not found. Skipping answer extraction.")
        final_answer = None
    else:
        final_answer = answer_match.group(1).strip()

    return reasoning, final_answer


def extract_model_answer_from_extracted(sample: str) -> Optional[str]:
    """
    Extract the model's final answer from the 'Extracted Answer' section.

    Args:
        sample: The raw log sample.

    Returns:
        The extracted answer or None if not found.
    """
    # Match the 'Extracted Answer' line
    extracted_answer_pattern = re.compile(r"Extracted Answer:\s*([A-Z])")
    match = extracted_answer_pattern.search(sample)
    if match:
        return match.group(1).strip()
    else:
        print("[Warning] 'Extracted Answer' not found in sample.")
        return None


def parse_log(log_content: str) -> Tuple[List[Dict], int, int]:
    """
    Parse the log content and extract structured data.

    Returns:
        parsed_data: A list of parsed log entries.
        total_instances: Total number of instances in the log.
        invalid_instances: Number of invalid instances in the log.
    """
    # Pattern to split into samples
    sample_pattern = re.compile(r"=+\n=+ Processing New Sample =+\n")

    # Patterns to extract specific sections
    ground_truth_pattern = re.compile(
        r"\[Ground Truth\] Correct Label: (.*?)\nOptions: (.*?)\n", re.S
    )
    final_score_pattern = re.compile(
        r"Final Score.*?Format: (.*?)\n.*?Answer: (.*?)\n.*?Total: (.*?)\n", re.S
    )

    # Split log into samples
    samples = sample_pattern.split(log_content)
    parsed_data = []

    # Track invalid instances
    invalid_instances = 0

    # Iterate through each sample
    for idx, sample in enumerate(samples[1:], start=1):  # Skip the first split as it will be empty
        parsed_sample = {}

        # Set default epoch and step
        parsed_sample["epoch"] = 0
        parsed_sample["step"] = 0

        # Extract Ground Truth
        ground_truth_match = ground_truth_pattern.search(sample)
        if ground_truth_match:
            correct_label = ground_truth_match.group(1).strip()
            options = ground_truth_match.group(2).strip()
            parsed_sample["ground_truth"] = {
                "correct_label": correct_label,
                "options": options,
            }
        else:
            parsed_sample["ground_truth"] = None  # Default when ground truth is missing

        # Extract Model Think and Answer
        reasoning, final_answer = extract_solution(sample)
        parsed_sample["model_think"] = reasoning

        # Try to get answer from "Extracted Answer" section first
        extracted_answer = extract_model_answer_from_extracted(sample)
        if extracted_answer:
            parsed_sample["model_answer"] = extracted_answer
        else:
            # Fallback to the <answer> tag parsing
            parsed_sample["model_answer"] = final_answer

        # Check if the instance is invalid
        if parsed_sample["model_think"] is None or parsed_sample["model_answer"] is None:
            invalid_instances += 1

        # Extract Final Score
        final_score_match = final_score_pattern.search(sample)
        if final_score_match:
            parsed_sample["final_score"] = {
                "format": float(final_score_match.group(1).strip()),
                "answer": float(final_score_match.group(2).strip()),
                "total": float(final_score_match.group(3).strip()),
            }
        else:
            parsed_sample["final_score"] = None  # Default when final score is missing

        parsed_data.append(parsed_sample)

    total_instances = len(parsed_data)
    return parsed_data, total_instances, invalid_instances


def save_to_json(data, output_file):
    """Save the parsed results to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)


def main():
    # Use argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="Parse log files and generate a JSON file.")
    parser.add_argument("log_file", help="Path to the log file.")
    parser.add_argument("-o", "--output", default=OUTPUT_FILE, help="Path to the output JSON file.")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Log file {args.log_file} does not exist!")
        return

    with open(log_path, "r") as f:
        log_content = f.read()

    # Parse the log content
    parsed_data, total_instances, invalid_instances = parse_log(log_content)
    valid_instances = total_instances - invalid_instances

    # Save to JSON
    save_to_json(parsed_data, args.output)
    print(f"Parsing completed! Results saved to {args.output}")

    # Print statistics
    print(f"\nStatistics:")
    print(f"- Total instances: {total_instances}")
    print(f"- Invalid instances: {invalid_instances}")
    print(f"- Valid instances: {valid_instances}")


if __name__ == "__main__":
    main()