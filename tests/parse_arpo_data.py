import re
import json
import srsly
import random
from pprint import pprint

TAG_PATTERN = re.compile(
    r"<(?P<tag>think|search|python|result|answer)>\s*(?P<content>.*?)\s*</\1>",
    re.DOTALL | re.IGNORECASE
)

def normalize_tool_name(tag: str) -> str:
    """
    把标签映射到 function 名称
    """
    t = tag.lower()
    if t == "search":
        return "search"
    if t == "python":
        return "python"
    return ""

def parse_to_messages(prompt: str, output: str):
    """
    把 prompt 和带标签的 output 转成 messages 列表
    """
    blocks = list(TAG_PATTERN.finditer(output))
    messages = []
    if prompt is not None:
        messages.append({"role": "user", "content": prompt.strip()})
    call_counter = 1
    last_call_id = None
    last_tool_name = None

    for m in blocks:
        tag = m.group("tag").lower()
        content = m.group("content").strip()

        if tag in ("search", "python"):
            tool_name = normalize_tool_name(tag)
            call_id = f"call_{call_counter}"
            call_counter += 1
            if tag == "search":
                args = {"query_list": [content]}
            elif tag == "python":
                args = {"code": content}

            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(args, ensure_ascii=False)
                        }
                    }
                ]
            })
            last_call_id = call_id
            last_tool_name = tool_name

        elif tag == "result":
            if last_call_id is None:
                # 没有 tool 调用的结果，直接跳过或作为普通 assistant
                continue
            messages.append({
                "role": "tool",
                "tool_call_id": last_call_id,
                "content": content
                # "content": json.dumps({"output": content}, ensure_ascii=False)
            })
            last_call_id = None
            last_tool_name = None

        elif tag == "answer":
            messages.append({
                "role": "assistant",
                "content": content
            })

    return messages

# ---------------- 示例 ----------------
if __name__ == "__main__":
    prompt = """You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool and python interpreter tool."""
    output = """<think> Okay, let's tackle this question. ... </think>
<search>Asharani BJP MLA 2013 suicide sentence</search>
<result>Asharani, a BJP MLA from Kerala, ... sentenced to **ten years of rigorous imprisonment** ...</result>
<think> ...more reasoning... </think>
<answer>\\boxed{10}</answer>"""

    data = list(srsly.read_jsonl("/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/local/ARPO-SFT-54K/final_5w4_still.jsonl"))

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/fs-computility/mabasic/shared/models/Qwen3-4B")

    for sample in random.sample(data, 10):
        assert len(sample["conversations"]) == 2
        prompt = sample["conversations"][0]["value"]
        output = sample["conversations"][1]["value"]
        msgs = parse_to_messages(prompt, output)
        print(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        print()
        print()
        print()
        print()
