from openai import OpenAI
import os
import math
import asyncio
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession, ListToolsResult, Tool

async def convert_mcp_to_openai_tools(mcp_tools: list) -> list:
    openai_tools = []

    for tool in mcp_tools.tools:
        tool_schema = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {}
            }
        }

        input_schema = tool.inputSchema

        parameters = {
            "type": input_schema['type'],
            "properties": input_schema['properties'],
            "required": input_schema['required'],
            "additionalProperties": False
        }

        for prop in parameters["properties"].values():
            if "enum" in prop:
                prop["description"] = f"Optional: {', '.join(prop['enum'])}"

        tool_schema["function"]["parameters"] = parameters
        openai_tools.append(tool_schema)

    # print("\nconverte to openai tools success:", [openai_tools])
    return openai_tools

async def main():
    # Connect to a streamable HTTP server
    async with streamablehttp_client("http://101.6.64.254:15555/mcp/") as (
        read_stream,
        write_stream,
        _,
    ):
        # Create a session using the client streams
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()
            # Call a tool
            mcp_tools = await session.list_tools()

            openai_tools = await convert_mcp_to_openai_tools(mcp_tools)
            for tool in openai_tools:
                print(tool)

            client = OpenAI(
                api_key="sk-353a88a777bd4c598f17b2923677e100",
                base_url="https://api.deepseek.com/v1"
                # base_url="http://localhost:8000/v1",
                # api_key="token-abc123",
            )

            completion = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "user", "content": "what's the combined headcount of the FAANG companies in 2024"}
                ],
                tools=openai_tools,
            )

            print(completion.choices[0].message)

# asyncio.run(main())

import requests
import json

url = "http://101.6.64.188:10086/run_code"

payload = json.dumps({
  "compile_timeout": 10,
  "run_timeout": 10,
  "code": """# cost of items
jacket_cost = 120
shoes_cost = 100 * 2 # since she wants 2 pairs of shoes
# total cost of items
total_cost = jacket_cost + shoes_cost
# earnings from babysitting
babysitting_earning = 5 * 10 # $5 each time, done 10 times
# initial money
initial_money = 100
# total money after babysitting
total_money_after_babysitting = initial_money + babysitting_earning
# amount still needed
amount_needed = total_cost - total_money_after_babysitting
# how much sara earns from mowing each time
mowing_earning = 10
# calculate how many times she needs to mow the lawn
num_times_mowing = amount_needed // mowing_earning # using integer division for whole number of times
# print result
print(num_times_mowing)""",
 "stdin": "string",
  "language": "python",
  "files": {},
  "fetch_files": [
    "string"
  ]
})
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)