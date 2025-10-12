import srsly

data = srsly.read_jsonl("/mnt/public/kyzhang/train.jsonl")

outputs = []
count = 0
for sample in data:
    sample["reward_model"] = sample["label"]
    del sample["label"]
    sample["extra_info"] = {"split":"train","index":count,"reference":None ,"public_test_cases":[{"input":"0","output":"0","testtype":"stdin"}],"private_test_cases": [{"input":"0","output":"0","testtype":"stdin"}],"difficulty":"","platform":"","func_name":""}
    outputs.append(sample)

srsly.write_jsonl("/mnt/public/kyzhang/MARTI/local/deepcoder/train.jsonl", outputs)