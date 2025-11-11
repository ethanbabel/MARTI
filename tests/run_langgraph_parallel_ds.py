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
import random
from pprint import pprint
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MAX_CONCURRENT_AGENTS = 2  # Adjust based on your system resources
TIMEOUT_PER_TASK = 600  # seconds

model = ChatOpenAI(
    # model="/fs-computility/mabasic/shared/models/Qwen3-32B",
    temperature=0.7,
    ##############
    model="kimi-k2-0711-preview",
    api_key="",
    base_url="https://api.moonshot.cn/v1",
    # model="deepseek-chat",
    # api_key="",
    # base_url="https://api.deepseek.com/v1"
    ##############
    # model="o3-mini",
    # api_key = "",
    # base_url = "https://lonlie.plus7.plus/v1",
    ##############
    # model="gpt-4.1",
    # api_key="",
    # base_url="https://api.sttai.cc/v1"
    # api_key="token-abc123",
    # base_url="http://localhost:8000/v1"
)

output_dir = "/mnt/public/kyzhang/MARTI/local/GAIA/outputs/kimi-k2-0719-preview-temp0.7"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

client = MultiServerMCPClient(
    {
        "gaia": {
            # Ensure your start your weather server on port 8000
            # "url": "http://101.6.65.240:15555/sse",
            # "transport": "sse",
            "url": "http://101.6.64.254:15555/mcp/",
            "transport": "streamable_http",
        }
    }
)

data = list(srsly.read_jsonl("./MARTI/local/GAIA/2023/validation/metadata.jsonl"))

async def process_single_sample(agent, sample: Dict[str, Any], sid: int, prompt_template: str, semaphore: asyncio.Semaphore) -> None:
    """Process a single sample with the agent"""
    async with semaphore:  # Acquire semaphore to limit concurrency
        task_id = sample.get('task_id')
        filename = os.path.join(output_dir, f"{task_id}---{sid}.json")
        
        if os.path.exists(filename):
            logger.info(f"Skipping {filename}, already exists.")
            return
        
        logger.info(f"Processing sample {task_id}---{sid}")
        
        try:
            # Configure recursive limits and other parameters
            config = {
                "recursion_limit": 25,  # Increase recursion limit
                "max_execution_time": TIMEOUT_PER_TASK,  # Maximum Execution Time (seconds)
            }
            
            # Add timeout to prevent hanging
            response = await asyncio.wait_for(
                agent.ainvoke(
                    {
                        "messages": [
                            {
                                "role": "user", 
                                "content": prompt_template.format(
                                    question=sample["Question"], 
                                    attachments="No attachments" if len(sample.get("file_name", [])) == 0 else sample.get("file_name", [])
                                )
                            },
                        ]
                    },
                    config=config,
                    debug=False  # Set to False for parallel execution to reduce output clutter
                ),
                timeout=TIMEOUT_PER_TASK
            )

            logger.info(f"Successfully processed sample {task_id}---{sid}")
            
            # Save results
            result_sample = sample.copy()
            result_sample["output"] = [msg.model_dump() for msg in response["messages"]]
            srsly.write_json(filename, result_sample)
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout processing sample {task_id}---{sid}")
            error_sample = sample.copy()
            error_sample["error"] = f"TimeoutError: Task exceeded {TIMEOUT_PER_TASK} seconds"
            error_sample["error_type"] = "timeout"
            srsly.write_json(filename.replace('.json', '_error.json'), error_sample)
            
        except GraphRecursionError as e:
            logger.error(f"GraphRecursionError processing sample {task_id}---{sid}: {e}")
            error_sample = sample.copy()
            error_sample["error"] = f"GraphRecursionError: {str(e)}"
            error_sample["error_type"] = "recursion_limit"
            srsly.write_json(filename.replace('.json', '_error.json'), error_sample)

        except KeyError as e:
            logger.error(f"KeyError processing sample {task_id}---{sid}: Missing key {e}")
            error_sample = sample.copy()
            error_sample["error"] = f"KeyError: {str(e)}"
            error_sample["error_type"] = "missing_key"
            srsly.write_json(filename.replace('.json', '_error.json'), error_sample)
        
        except Exception as e:
            logger.error(f"Unexpected error processing sample {task_id}---{sid}: {type(e).__name__}: {e}")
            error_sample = sample.copy()
            error_sample["error"] = f"{type(e).__name__}: {str(e)}"
            error_sample["error_type"] = "general_error"
            srsly.write_json(filename.replace('.json', '_error.json'), error_sample)

async def process_batch(agent, samples: List[Dict[str, Any]], sid: int, prompt_template: str, semaphore: asyncio.Semaphore) -> None:
    """Process a batch of samples in parallel"""
    tasks = []
    for sample in samples:
        task = asyncio.create_task(
            process_single_sample(agent, sample, sid, prompt_template, semaphore)
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    await asyncio.gather(*tasks, return_exceptions=True)

async def main():
    tools = await client.get_tools()
    
    agent = create_react_agent(
        model,
        tools,
    )

    # GAIA paper prompt
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

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_AGENTS)
    
    # Process all samples in parallel (with concurrency limit)
    for sid in range(32):
        logger.info(f"Starting batch processing for sid={sid} with {len(data)} samples")
        start_time = time.time()
        
        # Option 1: Process all samples at once (with semaphore limiting concurrency)
        await process_batch(agent, data, sid, prompt, semaphore)
        
        # Option 2: Process in smaller chunks for better control
        # chunk_size = 10
        # for i in range(0, len(data), chunk_size):
        #     chunk = data[i:i+chunk_size]
        #     logger.info(f"Processing chunk {i//chunk_size + 1}/{(len(data) + chunk_size - 1)//chunk_size}")
        #     await process_batch(agent, chunk, sid, prompt, semaphore)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Completed batch processing for sid={sid} in {elapsed_time:.2f} seconds")

    logger.info("All processing completed!")

if __name__ == "__main__":
    asyncio.run(main())
