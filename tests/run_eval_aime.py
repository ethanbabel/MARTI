import json
import srsly
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
    ReasoningEffort
)
 
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
 
# --- 1) Render the prefill with Harmony ---
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
 
# Harmony stop tokens (pass to sampler so they won't be included in output)
stop_token_ids = encoding.stop_tokens_for_assistant_actions()

# --- 2) Run vLLM with prefill ---
model_path = "/mnt/public/kyzhang/openai/gpt-oss-20b"
llm = LLM(
    model=model_path,
    trust_remote_code=True,
    tensor_parallel_size=2,
    max_seq_len_to_capture=16384*2
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

sampling = SamplingParams(
    max_tokens=16384*2,
    temperature=1,
    stop_token_ids=stop_token_ids,
)


# for effort in [ReasoningEffort.LOW, ReasoningEffort.MEDIUM, ReasoningEffort.HIGH]:
for effort in [ReasoningEffort.HIGH]:
    data = list(srsly.read_jsonl("/mnt/public/kyzhang/MARTI/data/AIME/test.jsonl"))
    for sample in data:
        convo = Conversation.from_messages(
            [
                Message.from_role_and_content(Role.SYSTEM, SystemContent.new().with_reasoning_effort(effort)),
                Message.from_role_and_content(
                    Role.DEVELOPER,
                    DeveloperContent.new().with_instructions("Please reason step by step, and put your final answer within \\boxed{}."),
                ),
                Message.from_role_and_content(Role.USER, sample["prompt"]),
            ]
        )

        prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

        outputs = llm.generate(
            prompt_token_ids=[prefill_ids],   # batch of size 1
            sampling_params=sampling,
        )

        # vLLM gives you both text and token IDs
        gen = outputs[0].outputs[0]
        text = gen.text
        output_tokens = gen.token_ids  # <-- these are the completion token IDs (no prefill)
        
        try:
            # --- 3) Parse the completion token IDs back into structured Harmony messages ---
            entries = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
            # 'entries' is a sequence of structured conversation entries (assistant messages, tool calls, etc.).
            print(entries)
            
            outputs = [json.dumps(message.to_dict()) for message in entries]
        except:
            outputs = None

        sample["outputs"] = outputs
        
    srsly.write_json("/mnt/public/kyzhang/MARTI/outputs/gpt-oss-20b/aime_16k_%s.json" % effort, data)
        