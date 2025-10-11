import srsly

# data = srsly.read_jsonl("/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/local/GAIA/2023/validation/metadata.jsonl")

# outputs = []
# for sample in data:
#     attachments = sample['file_name'] if len(sample['file_name']) > 0 else "No attachments."
#     outputs.append({
#         "problem": sample["Question"] + f"\n\nAttachments: {attachments}",
#         "answer": sample["Final answer"]
#     })

# srsly.write_json("/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/local/gaia-val/train.json", outputs)
# srsly.write_json("/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/local/gaia-val/test.json", outputs)

# data = srsly.read_jsonl("/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/local/webshaper.500.jsonl")

# outputs = []
# for sample in data:
#     outputs.append({
#         "problem": sample["question"] + "\n\nAttachments: No attachments.",
#         "answer": sample["answer"]
#     })

# srsly.write_json("/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/local/webshaper/train.json", outputs)

# import pandas as pd
# # 读取 parquet 文件
# # df = pd.read_parquet('/fs-computility/mabasic/qibiqing/marti-project/hotpotqa/hotpotqa_dev.parquet')
# df = pd.read_parquet("/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/local/ARPO-RL-Reasoning-10K/train_10k.parquet")
# # 显示前几行数据
# print(df.head())
# # 可选：查看所有字段名
# print(df.columns)
# outputs = []
# for row in df.iterrows():
#     print(row[1]["data_source"])
#     print(row[1]["prompt"])
#     print(row[1]["ability"])
#     print(row[1]["reward_model"])
#     print(row[1]["extra_info"])
#     print(row[1]["metadata"])
#     # print(row[1]["context"].split("\n\n"))
#     print()
#     print()
#     outputs.append({
#         "question": row[1]["prompt"][-1]["content"],
#         "answer": row[1]["reward_model"]["ground_truth"],
#         "metadata": {
#             "source": "search" if row[1]["ability"] == "qa" else "math"
#         }
#     })

# import random
# print(len(outputs))
# srsly.write_jsonl("/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/local/agent-arpo/train.jsonl", outputs)
# srsly.write_jsonl("/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/local/agent-arpo/test.jsonl", random.sample(outputs, 100))

# """
# [{'content': 'You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool and python interpreter tool. Given a question, you need to first think about the reasoning process in the mind and then provide the answer. During thinking, you can invoke the wikipedia search tool to search and python interpreter tool to calculate the math problem for fact information about specific topics if needed. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> <think> This is the reasoning process. </think> <python> python code here </python> <result> python interpreter result here </result> <think> This is the reasoning process. </think> <answer> The final answer is \\[ \x08oxed{answer here} \\] </answer>. In the last part of the answer, the final exact answer is enclosed within \x08oxed{} with latex format.', 'role': 'system'}
#  {'content': 'The sum of the digits of a ten-digit number is four. What can be the sum of the digits of the square of this number?', 'role': 'user'}]
# math
# {'ground_truth': '16', 'style': 'rule'}
# {'index': 9997, 'need_tools_kwargs': True, 'question': 'The sum of the digits of a ten-digit number is four. What can be the sum of the digits of the square of this number?', 'split': 'train'}
# """



# import srsly
# for task in ["2wiki", "SimpleQA", "aime24", "aime25", "bamboogle", "hotpotqa", "math500", "musique"]:
#     data = list(srsly.read_jsonl(f"/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/local/ARPO-main/evaluation/data/{task}/test.jsonl"))
#     print(task, data[0].keys(), type(data[0]["answer"]))
#     if task in ["aime24", "aime25"]:
#         outputs = [
#             {"question": sample["problem"], "answer": str(sample["answer"]), "metadata": {"source": "math"}}
#             for sample in data
#         ]
#     else:
#         outputs = [
#             {"question": sample["question"], "answer": str(sample["answer"]) if not isinstance(sample["answer"], list) else "LABEL_SEP".join(sample["answer"]), 
#             "metadata": {"source": "math" if task == "math500" else "search"}}
#             for sample in data
#         ]
#     srsly.write_jsonl("/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/local/agent-bench/%s.jsonl" % task, outputs)
import os
import srsly
import json
import sys
from transformers import AutoTokenizer

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from marti.worlds.tools.parser import ToolParser

# tool_parser = ToolParser()
# # results = tool_parser.parse_tools("<tool_call>\n{\"name\": \"python\", \"arguments\": \"{\\\"code\\\": \\\"import sympy as sp\\\\r\\\\n\\\\r\\\\n# Define the variable\\\\r\\\\nx = sp.symbols('x')\\\\r\\\\n\\\\r\\\\n# Define the function f(x)\\\\r\\\\nf = sp.sqrt(3) * sp.sin(x) * sp.cos(x) - (1/2) * sp.cos(2*x)\\\\r\\\\n\\\\r\\\\n# Simplify the function\\\\r\\\\nf_simplified = sp.simplify(f)\\\\r\\\\nprint(f_simplified)\\\"}\"}\n</tool_call>")
# results = tool_parser.parse_tools( "<tool_call>\n{\"name\": \"search\", \"arguments\": {\"query_list\": [\"Houston Academy For International Studies location\"]}}\n</tool_call><|im_end|>")
# for result in results:
#     args_dict = json.loads(result.get("args", "{}"))
#     print(args_dict, type(args_dict))
# tokenizer = AutoTokenizer.from_pretrained("/fs-computility/mabasic/shared/models/Qwen3-4B-Instruct-2507")
# tokenizer = AutoTokenizer.from_pretrained("/fs-computility/mabasic/shared/models/Qwen2.5-3B-Instruct")
# data = srsly.read_jsonl("/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/local/agent-sft/test.jsonl")
# for sample in data:
#     inputs = tokenizer.apply_chat_template(sample["messages"][:3], tokenize=False, add_generation_prompt=False)
#     print(inputs)
#     print(tool_parser.parse_tools(inputs))
#     print()
#     print()
#     print()

# for file in ["local/agent-arpo/train.jsonl",
#     "local/agent-arpo/test.jsonl",
#     "local/agent-bench/2wiki.jsonl",
#     "local/agent-bench/aime24.jsonl",
#     "local/agent-bench/aime25.jsonl",
#     "local/agent-bench/bamboogle.jsonl",
#     "local/agent-bench/hotpotqa.jsonl",
#     "local/agent-bench/math500.jsonl",
#     "local/agent-bench/musique.jsonl",
#     "local/agent-bench/SimpleQA.jsonl"]:
#     data = list(srsly.read_jsonl("/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/" + file))
#     for sample in data:
#         sample["problem"] = sample["question"]
#     srsly.write_jsonl("/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/" + file, data)

# for task in ["2wiki", "bamboogle", "hotpotqa", "musique"]:
# for task in ["aime24", "aime25", "math500"]:
#     data = list(srsly.read_jsonl("/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/local/agent-bench/%s.jsonl" % task))
#     for sample in data:
#         del sample["metadata"]

#     srsly.write_jsonl("/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/local/search-r1-bench-lite/%s.jsonl" % task, data)

for split in ["train", "test"]:
    data = srsly.read_json("/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/data/MATH/%s.json" % split)
    for sample in data:
        sample["problem"] = sample["prompt"]
    srsly.write_json("/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/local/math/%s.json" % split, data)