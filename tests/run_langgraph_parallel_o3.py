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

# Modified model configuration to support reasoning content
model = ChatOpenAI(
    model="o3-mini",
    api_key = "sk-njeP9itAfJioZsxXSlGHpQQvDr8fzWPceIBC772EL0jxw9yQ",
    base_url = "https://lonlie.plus7.plus/v1",
    temperature=0.0,
    # Critical: Enable Responses API to get reasoning content
    use_responses_api=True,
    # Configure reasoning parameters
    # model_kwargs={
    #     "reasoning": {
    #         "effort": "medium",  # Options: "low", "medium", "high"
    #         "summary": "auto"    # Options: "auto", "concise", "detailed", or None
    #     }
    # }
    # Alternative: Use new output format (langchain-openai >= 0.3.26)
    # output_version="responses/v1"  # This puts reasoning in content field instead of additional_kwargs
)

output_dir = "/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/local/GAIA/outputs/o3-mini-new"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

client = MultiServerMCPClient(
    {
        "gaia": {
            "url": "http://101.6.64.254:15555/mcp/",
            "transport": "streamable_http",
        }
    }
)

data = list(srsly.read_jsonl("/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/local/GAIA/2023/validation/metadata.jsonl"))[:3]

def extract_reasoning_content(msg):
    """Extract reasoning content from message"""
    reasoning_data = {}
    
    # Get reasoning from additional_kwargs
    if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
        # Full reasoning summary
        if 'reasoning' in msg.additional_kwargs:
            reasoning_data['reasoning'] = msg.additional_kwargs['reasoning']
        
        # Streaming reasoning chunks
        if 'reasoning_summary_chunk' in msg.additional_kwargs:
            reasoning_data['reasoning_summary_chunk'] = msg.additional_kwargs['reasoning_summary_chunk']
        
        if 'reasoning_summary' in msg.additional_kwargs:
            reasoning_data['reasoning_summary'] = msg.additional_kwargs['reasoning_summary']
    
    return reasoning_data

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
            
            # Enhanced save results with reasoning content extraction
            result_sample = sample.copy()
            
            # Save all messages with their reasoning content
            enhanced_messages = []
            for msg in response["messages"]:
                msg_dict = msg.model_dump()
                
                # Extract and add reasoning content explicitly
                reasoning_content = extract_reasoning_content(msg)
                if reasoning_content:
                    msg_dict['extracted_reasoning'] = reasoning_content
                    logger.info(f"Found reasoning content for message: {list(reasoning_content.keys())}")
                
                enhanced_messages.append(msg_dict)
            
            result_sample["output"] = enhanced_messages
            
            # Optionally, save reasoning content summary separately
            reasoning_summary = []
            for i, msg in enumerate(response["messages"]):
                reasoning = extract_reasoning_content(msg)
                if reasoning:
                    reasoning_summary.append({
                        "message_index": i,
                        "reasoning_content": reasoning
                    })
            
            if reasoning_summary:
                result_sample["reasoning_summary"] = reasoning_summary
            
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
    for sid in range(1):
        logger.info(f"Starting batch processing for sid={sid} with {len(data)} samples")
        start_time = time.time()
        
        await process_batch(agent, data, sid, prompt, semaphore)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Completed batch processing for sid={sid} in {elapsed_time:.2f} seconds")

    logger.info("All processing completed!")

if __name__ == "__main__":
    asyncio.run(main())