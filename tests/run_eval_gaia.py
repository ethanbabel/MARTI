import os
import re
import string
import logging
from typing import Any, Optional, List

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "WARN"))

def question_scorer(model_answer: str, ground_truth: str) -> bool:
    r"""Scorer for the GAIA benchmark.
    https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/
    scorer.py

    Args:
        model_answer (str): The model answer.
        ground_truth (str): The ground truth answer.

    Returns:
        bool: The score of the model
    """

    def is_float(element: Any) -> bool:
        try:
            float(element)
            return True
        except ValueError:
            return False

    if is_float(ground_truth):
        logger.info(f"Evaluating {model_answer} as a number.")
        normalized_answer = normalize_number_str(model_answer)
        return normalized_answer == float(ground_truth)

    elif any(char in ground_truth for char in [",", ";"]):
        logger.info(
            f"Evaluating {model_answer} as a comma separated list."
        )
        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)

        if len(gt_elems) != len(ma_elems):
            logger.warning(
                "Answer lists have different lengths, returning False.",
                UserWarning,
            )
            return False

        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                comparisons.append(normalized_ma_elem == float(gt_elem))
            else:
                ma_elem = normalize_str(ma_elem, remove_punct=False)
                gt_elem = normalize_str(gt_elem, remove_punct=False)
                comparisons.append(ma_elem == gt_elem)
        return all(comparisons)
    else:
        logger.info(f"Evaluating {model_answer} as a string.")
        ma_elem = normalize_str(model_answer)
        gt_elem = normalize_str(ground_truth)
        return ma_elem == gt_elem


def normalize_number_str(number_str: str) -> float:
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        logger.error(
            f"String {number_str} cannot be normalized to number str."
        )
        return float("inf")


def split_string(
    s: str, char_list: Optional[List[str]] = None
) -> list[str]:
    r"""Split a string based on a list of characters.

    Args:
        s (str): The string to split.
        char_list (Optional[List[str]], optional): T
            he list of characters to split on.
            (default: :obj:`None`)
    """
    if char_list is None:
        char_list = [",", ";"]
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def normalize_str(input_str, remove_punct=True) -> str:
    r"""Normalize a string.

    Args:
        input_str: The input string to normalize.
        remove_punct: Whether to remove punctuation.

    Returns:
        str: The normalized string.
    """
    no_spaces = re.sub(r"\s", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    else:
        return no_spaces.lower()


if __name__ == "__main__":
    import os
    import srsly
    import numpy as np
    import random
    from collections import defaultdict, Counter
    # data = srsly.read_jsonl("/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/local/GAIA/2023/validation/metadata.jsonl")
    data = srsly.read_jsonl("/mnt/public/kyzhang/MARTI/local/GAIA/2023/validation/metadata.jsonl")
    id2sample = {d["task_id"]: d for d in data}

    id2answers = defaultdict(list)
    # data_dir = "/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/local/GAIA/outputs/gpt-4.1-temp0.7"
    # data_dir = "/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/local/GAIA/outputs/deepseek-chat-temp0.7"
    # data_dir = "/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/local/GAIA/outputs/Qwen3-32B-fix"
    data_dir = "/mnt/public/kyzhang/MARTI/local/GAIA/outputs/kimi-k2-0719-preview-temp0.7"
    for file in os.listdir(data_dir):
        if "error" in file:
            continue

        sample = srsly.read_json(os.path.join(data_dir, file))
        answer = sample["output"][-1]["content"].split("<answer>")[-1].split("</answer>")[0].strip()

        id2answers[sample["task_id"]].append([answer, file.split("---")[-1].split(".")[0]])
    
    for idx, answers in id2answers.items():
        answers = sorted(answers, key=lambda item:item[1])
        answers = [ans[0] for ans in answers]
        id2answers[idx] = answers

    result = {
        "maj": [],
        "pass": []
    }

    for n in [1, 2, 4, 8, 16, 32]:
        list_acc = []
        list_pass = []
        for _ in range(3):
            all_acc = []
            pass_n = []
            for idx, answers in id2answers.items():
                ground_truth = id2sample[idx]["Final answer"]
                # answers = random.sample(answers, min(len(answers), n))
                answers = answers[:n]
                counts = Counter(answers)
                majority_answer , _ = counts.most_common(1)[0]
                all_acc.append(question_scorer(majority_answer, ground_truth))

                pass_n.append(any([question_scorer(answer, ground_truth) for answer in answers]))
            
            avg_pass_n = np.mean(pass_n)
            avg_all_acc = np.mean(all_acc)
            list_acc.append(avg_all_acc)
            list_pass.append(avg_pass_n)

        avg_pass_n = np.mean(list_pass)
        avg_all_acc = np.mean(list_acc)
        print(f"Pass@{n}", avg_pass_n)
        print(f"Maj@{n}", np.mean(avg_all_acc))
        print()

        result["maj"].append(np.mean(avg_all_acc))
        result["pass"].append(avg_pass_n)
    
    from pprint import pprint
    pprint(result)

"""
Qwen3-32B, temperature=0.0
{'maj': [0.2138364779874214,
         0.20754716981132076,
         0.2138364779874214,
         0.22641509433962265,
         0.22641509433962265],
 'pass': [0.2138364779874214,
          0.23270440251572327,
          0.23270440251572327,
          0.2641509433962264,
          0.2830188679245283]}

Qwen3-32B, temperature=0.7
{'maj': [0.1393939393939394,
         0.14545454545454545,
         0.1393939393939394,
         0.15151515151515152,
         0.17575757575757575],
 'pass': [0.1393939393939394,
          0.2,
          0.23030303030303031,
          0.3333333333333333,
          0.4]}

Qwen3-8B, temperature=0.7
{'maj': [0.10062893081761007,
         0.12578616352201258,
         0.13836477987421383,
         0.14465408805031446,
         0.13836477987421383],
 'pass': [0.10062893081761007,
          0.15723270440251572,
          0.15723270440251572,
          0.15723270440251572,
          0.15723270440251572]}

Qwen3-4B, temperature=0.7
{'maj': [0.07878787878787878,
         0.10909090909090909,
         0.11515151515151516,
         0.12121212121212122,
         0.12727272727272726],
 'pass': [0.07878787878787878,
          0.15151515151515152,
          0.15757575757575756,
          0.15757575757575756,
          0.15757575757575756]}

deepseek-chat, temperature=0.7
{'maj': [0.12121212121212122,
         0.12727272727272726,
         0.10909090909090909,
         0.14545454545454545,
         0.12727272727272726],
 'pass': [0.12121212121212122,
          0.16363636363636364,
          0.18181818181818182,
          0.21212121212121213,
          0.2606060606060606]}
"""