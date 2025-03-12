import re
from typing import Dict, Optional

# List of possible assistant headers
assistant_headers = [
    "Assistant:",
    "<|im_start|>assistant",
    "<|start_header_id|>assistant<|end_header_id|>"
]

# List of possible EOS tokens
EOS_TOKENS = [
    "<|eom_id|>", 
    "<|eot_id|>", 
    "<|end_of_text|>", 
    "<|im_end|>", 
    "<|endoftext|>",
    ]

# Default reward/penalty values
default_reward_params = {
    "format_correct": 1,         # Reward for correct format
    "format_incorrect": -1,      # Penalty for incorrect format
    "answer_correct": 2,         # Reward for correct answer
    "answer_incorrect": -2,      # Penalty for incorrect answer
    "answer_invalid": -3,        # Penalty for invalid answer
}


def extract_solution(solution_str: str) -> Optional[str]:
    """
    Extracts the final answer from the model's response string.

    Args:
        solution_str: Raw response string from the language model.

    Returns:
        Extracted answer string or None if extraction fails.
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

    # Ensure <think> tags are present before <answer> tags
    if "<think>" not in solution_str or "</think>" not in solution_str:
        print("[Error] Missing <think> reasoning tags")
        return None
    if "<answer>" not in solution_str or "</answer>" not in solution_str:
        print("[Error] Missing <answer> tags")
        return None

    # Ensure <answer> comes after </think>
    think_end_idx = solution_str.find("</think>")
    answer_start_idx = solution_str.find("<answer>")
    if think_end_idx > answer_start_idx or think_end_idx == -1 or answer_start_idx == -1:
        print("[Error] <answer> tag appears before </think> tag")
        return None

    # Extract the final answer enclosed in <answer> tags
    # answer_pattern = r'<answer>\((.*?)\)\s*(.*?)</answer>'
    answer_pattern = r'<answer>\s*\(([A-Z])\)\s*(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    if match:
        final_answer = match.group(1).strip()
        return final_answer
    else:
        # No valid <answer> tags found
        print("[Error] No valid <answer> tags found.")
        return None


def validate_response_structure(processed_str: str) -> bool:
    """
    Validates whether the response structure meets the proper format.

    Args:
        processed_str: Processed response string from the model.

    Returns:
        Boolean indicating whether the structure is valid.
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Find the correct header
    matched_header = None
    start_index = -1
    for header in assistant_headers:
        if header in processed_str:
            start_index = processed_str.find(header)
            matched_header = header
            break

    if start_index != -1 and matched_header:
        # Extract the portion of the response after the header
        processed_str = processed_str[start_index + len(matched_header):].strip()
    else:
        print("[Error] Failed to locate model response header")
        return False

    # Check for required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = processed_str.find(tag_str)

        print(f"  {tag_str}: count={count}, position={positions[tag_name]}")
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Validate tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    # Check for interruptions between </think> and <answer>
    think_end_pos = positions['think_end']
    answer_start_pos = positions['answer_start']
    if think_end_pos != -1 and answer_start_pos != -1:
        between_content = processed_str[think_end_pos + len('</think>'):answer_start_pos].strip()
        if between_content:
            print(f"  [Error] Unexpected content between </think> and <answer>.")
            validation_passed = False
        else:
            print("  No extraneous content found between </think> and <answer>.")

    # Check for extraneous content after the first <answer>...</answer>
    answer_end_pos = positions['answer_end']
    if answer_end_pos != -1:
        after_answer_content = processed_str[answer_end_pos + len('</answer>'):].strip()
        # Check if the content after </answer> is in the list of EOS tokens
        if EOS_TOKENS and after_answer_content and after_answer_content not in EOS_TOKENS:
            print(f"  [Error] Unexpected content after </answer>: '{after_answer_content}'")
            validation_passed = False
        elif after_answer_content in EOS_TOKENS:
            print(f"  Valid EOS token found after </answer>: '{after_answer_content}'")
        else:
            print("  No extraneous content found after </answer>.")

    return validation_passed


def compute_score(
    solution_str: str,
    ground_truth: Dict[str, str],
    reward_params: Dict[str, float] = None
):
    """
    Computes a reward score for the Social IQa task based on model responses.

    Args:
        solution_str: Raw model response string.
        ground_truth: Dictionary containing ground truth answers.
        reward_params: Dictionary containing reward and penalty values.

    Returns:
        Total score (sum of format and answer rewards).
    """
    print("\n" + "=" * 80)
    print(" Processing New Sample ".center(80, '='))

    # Log the raw model output
    print("[Model Output]")
    print(solution_str)
    print("-" * 80)

    # Use provided reward_params or defaults
    reward_params = reward_params or default_reward_params

    # Map numeric labels ("1", "2", "3") to letter labels ("A", "B", "C")
    label_mapping = {"1": "A", "2": "B", "3": "C"}
    correct_label = label_mapping.get(ground_truth.get("label", ""), "")
    options = {
        "A": ground_truth.get("answerA"),
        "B": ground_truth.get("answerB"),
        "C": ground_truth.get("answerC"),
    }

    print(f"[Ground Truth] Correct Label: {correct_label}")
    print(f"Options: {options}")

    # Extract model's answer
    extracted_answer = extract_solution(solution_str)
    if extracted_answer:
        print(f"Extracted Answer: {extracted_answer}")
    else:
        print("[Error] Failed to extract model answer")
        extracted_answer = None  # Keep answer invalid, continue validation

    # Validate response structure
    format_valid = validate_response_structure(solution_str)
    format_score = reward_params["format_correct"] if format_valid else reward_params["format_incorrect"]
    print(f"Format Validation: {'PASS' if format_valid else 'FAIL'}")
    print(f"Format Score: {format_score}")

    # Check if extracted answer matches the ground truth
    answer_score = 0
    if extracted_answer not in options:
        print(f"[Error] Extracted answer '{extracted_answer}' is not a valid option")
        answer_score = reward_params["answer_invalid"]  # Penalty for invalid answers
    elif extracted_answer == correct_label:
        print(f"Answer Validation: PASS (Correct Answer)")
        answer_score = reward_params["answer_correct"]  # Reward for correct answer
    else:
        print(f"Answer Validation: FAIL (Incorrect Answer)")
        answer_score = reward_params["answer_incorrect"]  # Penalty for incorrect answers

    # Final score
    total_score = format_score + answer_score
    print("\n" + "-" * 80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer: {answer_score}")
    print(f"  Total: {total_score}")
    print("=" * 80 + "\n")

    return total_score