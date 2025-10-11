import asyncio
import srsly
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts import ChatPromptTemplate
from langgraph.errors import GraphRecursionError
import time, srsly
import os
import sys
import random
from pprint import pprint

model_name = sys.argv[1]

print(model_name)

assert model_name in ["Qwen3-4B", "Qwen3-8B", "Qwen2.5-7B-Instruct"]

model = ChatOpenAI(
    model=model_name,
    temperature=0.0,
    base_url="http://101.6.65.222:18000/v1",
    api_key="null",
)

output_dir = f"/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/local/GAIA/outputs/{model_name}"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

client = MultiServerMCPClient(
    {
        "gaia": {
            # Ensure your start your weather server on port 8000
            "url": "http://101.6.64.254:15555/mcp",
            "transport": "streamable_http",
        }
    }
)

data = list(srsly.read_jsonl("/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/local/GAIA/2023/validation/metadata.jsonl"))

async def main():
    tools = await client.get_tools()
    
    agent = create_react_agent(
        model,
        tools,
    )

    prompt_opt1 = """You are a general AI assistant equipped with a variety of tools. I will ask you a question. Start by reasoning through the problem and creating a step-by-step plan. Then, if necessary, use available tools one by one to gather or verify information. Make sure your reasoning and tool usage are clearly separated. 

You should finish your answer with the following tags: <answer> [YOUR FINAL ANSWER] </answer>  
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma-separated list of numbers and/or strings.

If you are asked for a number, do not use commas or units (like $ or %), unless explicitly specified.  
If you are asked for a string, do not use articles or abbreviations (e.g., for cities), and write digits in plain text unless specified otherwise.  
If you are asked for a comma-separated list, follow the above rules depending on the element type.

Question: Your final answer should be a number, or as few words as possible.  
{question}

Attachments: {attachments}"""

    prompt_opt2 = """You are a general AI assistant with access to tools. For each question, think step by step about what you need to do, then use tools as needed to find the answer.

Report your thoughts and finish your answer with: <answer>[YOUR FINAL ANSWER]</answer>

YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.

If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise.
If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.

Question: {question}

Attachments: {attachments}"""

    # GAIA paper
    prompt = """You are a general AI assistant. I will ask you a question. Report your thoughts, and finish
your answer with the following tags: <answer> [YOUR FINAL ANSWER] </answer>

YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of
numbers and/or strings.

If you are asked for a number, don't use comma to write your number neither use units such as $ or percent
sign unless specified otherwise.
If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in
plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put
in the list is a number or a string.

Question: Your final answer should be a number, or as few words as possible.
{question}

Attachments: {attachments}"""

    for sid in range(16):
        for sample in data:
            pprint(sample)

            task_id = sample.get('task_id')
            filename = os.path.join(output_dir, f"{task_id}---{sid}.json")

            if os.path.exists(filename):
                print(f"Skipping {filename}, already exists.")
                continue

            try:
                # # Configure recursive limits and other parameters
                # config = {
                #     "recursion_limit": 50,  # Increase recursion limit
                #     "max_execution_time": 300,  # Maximum Execution Time (seconds)
                # }
                
                response = await agent.ainvoke(
                    {
                        "messages": [
                            {
                                "role": "user", 
                                "content": prompt.format(
                                    question=sample["Question"], 
                                    attachments="No attachments" if len(sample.get("file_name", [])) == 0 else sample.get("file_name", [])
                                )
                            },
                        ]
                    },
                    # config=config,
                    debug=True
                )

                print(response["messages"][-1].content)
                print("\n"*5)
                
                sample["output"] = [msg.model_dump() for msg in response["messages"]]
                print(sample)
                srsly.write_json(filename, sample)
                
            except GraphRecursionError as e:
                print(f"GraphRecursionError processing sample {task_id}---{sid}: {e}")
                print("Consider increasing recursion_limit or simplifying the task.")
                # Record error but continue execution
                error_sample = sample.copy()
                error_sample["error"] = f"GraphRecursionError: {str(e)}"
                error_sample["error_type"] = "recursion_limit"
                srsly.write_json(filename.replace('.json', '_error.json'), error_sample)
                continue

            except KeyError as e:
                print(f"KeyError processing sample {task_id}---{sid}: Missing key {e}")
                print(f"Available keys in sample: {list(sample.keys())}")
                # Record error but continue execution
                error_sample = sample.copy()
                error_sample["error"] = f"KeyError: {str(e)}"
                error_sample["error_type"] = "missing_key"
                srsly.write_json(filename.replace('.json', '_error.json'), error_sample)
                continue
            
            except Exception as e:
                print(f"Unexpected error processing sample {task_id}---{sid}: {type(e).__name__}: {e}")
                # Record error but continue execution
                error_sample = sample.copy()
                error_sample["error"] = f"{type(e).__name__}: {str(e)}"
                error_sample["error_type"] = "general_error"
                srsly.write_json(filename.replace('.json', '_error.json'), error_sample)
                continue

asyncio.run(main())
