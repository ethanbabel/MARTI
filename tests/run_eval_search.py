import re

class TagMatch:
    """Helper class to store information about found tags."""
    def __init__(self, tag_type: str, content: str, start: int, end: int):
        self.type = tag_type  # 'search', 'information', 'answer'
        self.content = content.strip() # Content within the tag, stripped
        self.start = start # Start index of the full tag in the response
        self.end = end     # End index of the full tag in the response

    def __repr__(self):
        return f"TagMatch(type='{self.type}', content='{self.content[:20]}...', start={self.start}, end={self.end})"

def check_agent_response_pattern(response_text: str) -> int:
    """
    Checks if the agent's response strictly matches the specified pattern.

    Args:
        response_text: The agent's full response string.

    Returns:
        1 if the pattern is fully matched, 0 otherwise.
    """
    if not response_text: # Basic check for empty input
        return 0

    # Regex to find all occurrences of <search>, <information>, or <answer> tags
    # It captures the full tag and its inner content for each type.
    tag_regex = re.compile(
        r"(<search>(.*?)</search>)|"
        r"(<information>(.*?)</information>)|"
        r"(<answer>(.*?)</answer>)",
        re.DOTALL  # Allows '.' to match newlines, important for multi-line content
    )

    all_tags = []
    for match in tag_regex.finditer(response_text):
        start_idx, end_idx = match.span() # Get start and end of the whole matched tag
        tag_type = ''
        content = ''

        if match.group(1):  # Matched <search>...</search>
            tag_type = 'search'
            content = match.group(2)
        elif match.group(3):  # Matched <information>...</information>
            tag_type = 'information'
            content = match.group(4)
        elif match.group(5):  # Matched <answer>...</answer>
            tag_type = 'answer'
            content = match.group(6)
        
        if tag_type: # Should always be true if a match occurs
            all_tags.append(TagMatch(tag_type, content, start_idx, end_idx))

    # --- Rule 1: Basic Tag Structure and Presence ---
    if not all_tags:  # No tags found at all.
        return 0

    num_search_tags = sum(1 for t in all_tags if t.type == 'search')
    num_info_tags = sum(1 for t in all_tags if t.type == 'information')
    num_answer_tags = sum(1 for t in all_tags if t.type == 'answer')

    if num_search_tags == 0: # Must have at least one search operation.
        return 0
    if num_answer_tags != 1: # Must have exactly one answer tag.
        return 0
    if all_tags[-1].type != 'answer': # The very last tag must be an answer tag.
        return 0
    if num_search_tags != num_info_tags: # Each search must be paired with information.
        return 0

    # --- Rule 2: Tag Content ---
    # All tags must have non-empty content (already stripped in TagMatch constructor).
    for tag in all_tags:
        if not tag.content:
            return 0

    # --- Rule 3: Sequence of Tags ---
    # Expected sequence: S, I, S, I, ..., S, I, A
    if all_tags[0].type != 'search': # Must start with a search tag.
        return 0

    for i in range(len(all_tags) - 1):
        current_tag = all_tags[i]
        next_tag = all_tags[i+1]

        if current_tag.type == 'search':
            if next_tag.type != 'information':
                return 0  # Search must be followed by Information.
        elif current_tag.type == 'information':
            # Information can be followed by another Search or the final Answer.
            if not (next_tag.type == 'search' or next_tag.type == 'answer'):
                return 0
        # An 'answer' tag can only be at the end (covered by all_tags[-1].type check).
        # If current_tag.type is 'answer' and it's not the last tag, this loop structure handles it.

    # --- Rule 4: Thinking Processes ---
    # 4.1. Initial Thinking Process (before the first tag)
    initial_tp_text = response_text[0:all_tags[0].start].strip()
    if not initial_tp_text:
        return 0

    # 4.2. Intermediate Thinking Processes
    for i in range(len(all_tags) - 1):
        tag_i = all_tags[i]
        tag_i_plus_1 = all_tags[i+1]
        
        text_between_tags = response_text[tag_i.end:tag_i_plus_1.start].strip()

        if tag_i.type == 'search' and tag_i_plus_1.type == 'information':
            # There should be NO thinking process text between a <search> and its direct <information>
            if text_between_tags:
                return 0
        elif tag_i.type == 'information' and \
             (tag_i_plus_1.type == 'search' or tag_i_plus_1.type == 'answer'):
            # There MUST BE a thinking process between <information> and the next <search>
            # or between the last <information> and the <answer>.
            if not text_between_tags:
                return 0
        # Other tag sequences (e.g., S-S, I-I) would have been caught by Rule 3.

    # --- Rule 5: End of Response ---
    # The response must end immediately after the closing </answer> tag.
    # Any trailing characters (even whitespace, after stripping) means it doesn't fully match.
    if response_text[all_tags[-1].end:].strip():
        return 0
        
    # If all rules pass:
    return 1

# Example Usage with your provided text:
agent_response_example = """I need to search for Paul Walker's cars in Fast and Furious movies.
<search> Paul Walker's cars in Fast and Furious </search>
<information>Doc 1(Title: "Paul Walker") also had a large car collection of about 30 cars, a portion of which he co-owned along with Roger Rodas. The collection included (but is not limited to): Roger Rodas became Walker's financial adviser in 2007 and helped to establish Reach Out Worldwide. Rodas, a pro-am racer was the CEO of Always Evolving, a Valencia high-end vehicle performance shop owned by Walker. Walker was also a close friend of his ""2 Fast 2 Furious"" co-star Tyrese Gibson. Vin Diesel considered Walker to be like a brother, both on and off screen, and affectionately called him ""Pablo"". Walker's mother referred to
Doc 2(Title: "Paul Walker") Paul Walker Paul William Walker IV (September 12, 1973 – November 30, 2013) was an American actor best known for his role as Brian O'Conner in ""The Fast and the Furious"" franchise. Walker first gained prominence in 1999 with roles in the teen films ""She's All That"" and ""Varsity Blues"". In 2001, he gained international fame for his role in the street racing action film ""The Fast and the Furious"" (2001), a role he reprised in five of the next six installments, but died in 2013 in the middle of filming ""Furious 7"" (2015). Walker began his career guest-starring on
Doc 3(Title: "Paul Walker") of Porsche in a separate lawsuit filed by Roger Rodas' widow, Kristine. The ruling had no bearing on two other cases against Porsche which have been filed by Walker's father, who is also the executor of the actor's estate, and his daughter. Walker's father and daughter both reached an agreement with Porsche. Paul Walker Paul William Walker IV (September 12, 1973 – November 30, 2013) was an American actor best known for his role as Brian O'Conner in ""The Fast and the Furious"" franchise. Walker first gained prominence in 1999 with roles in the teen films ""She's All That"" and</information>

From the information provided, it's clear that Paul Walker was a part of the "Fast and Furious" series, but the specific list of cars is not mentioned. Since I lack this particular detail, I will call a search engine to get the specific list of cars Paul Walker drove in the "Fast and Furious" movies.
<search> list of cars paul walker drove in Fast and Furious </search>
<information>Doc 1(Title: "Paul Walker") also had a large car collection of about 30 cars, a portion of which he co-owned along with Roger Rodas. The collection included (but is not limited to): Roger Rodas became Walker's financial adviser in 2007 and helped to establish Reach Out Worldwide. Rodas, a pro-am racer was the CEO of Always Evolving, a Valencia high-end vehicle performance shop owned by Walker. Walker was also a close friend of his ""2 Fast 2 Furious"" co-star Tyrese Gibson. Vin Diesel considered Walker to be like a brother, both on and off screen, and affectionately called him ""Pablo"". Walker's mother referred to
Doc 2(Title: "Paul Walker") Paul Walker Paul William Walker IV (September 12, 1973 – November 30, 2013) was an American actor best known for his role as Brian O'Conner in ""The Fast and the Furious"" franchise. Walker first gained prominence in 1999 with roles in the teen films ""She's All That"" and ""Varsity Blues"". In 2001, he gained international fame for his role in the street racing action film ""The Fast and the Furious"" (2001), a role he reprised in five of the next six installments, but died in 2013 in the middle of filming ""Furious 7"" (2015). Walker began his career guest-starring on
Doc 3(Title: "Paul Walker") of Porsche in a separate lawsuit filed by Roger Rodas' widow, Kristine. The ruling had no bearing on two other cases against Porsche which have been filed by Walker's father, who is also the executor of the actor's estate, and his daughter. Walker's father and daughter both reached an agreement with Porsche. Paul Walker Paul William Walker IV (September 12, 1973 – November 30, 2013) was an American actor best known for his role as Brian O'Conner in ""The Fast and the Furious"" franchise. Walker first gained prominence in 1999 with roles in the teen films ""She's All That"" and</information>

Based on the updated information, it seems the focus was on his career, financials, and family. However, I am still missing the specific list of cars he drove in the "Fast and Furious" movies. Since it appears that the information might not be contained within the accessed documents, and I have no further search queries to make, I will provide an answer based on the details I have.
<answer> Charger </answer>"""


import srsly
from collections import Counter
import numpy as np
from tqdm import tqdm
from pprint import pprint
from qa_em import compute_score_em

file2acc = {}
for file in [
    # "/mnt/public/kyzhang/MARTI/tests/outputs/data_with_search_results_nq_qwen2.5-3b-it-tool-calling.jsonl",
    # "/mnt/public/kyzhang/MARTI/tests/outputs/data_with_search_results_nq_qwen2.5-3b-it-tool-calling0609-simper.jsonl",
    # "/mnt/public/kyzhang/MARTI/tests/outputs/data_with_search_results_nq_qwen2.5-3b-it-tool-calling0609-simper2.jsonl",
    # "/mnt/public/kyzhang/MARTI/tests/outputs/data_with_search_results_nq_qwen2.5-3b-it-tool-calling0609-wo-think.jsonl",
    # "/mnt/public/kyzhang/MARTI/tests/outputs/data_with_search_results_nq_qwen2.5-3b-it-tool-calling0609.jsonl",
    # "/mnt/public/kyzhang/MARTI/tests/outputs/data_with_search_results_nq_qwen2.5-3b-it-tool-calling0609-think2-all.jsonl",
    # "/mnt/public/kyzhang/MARTI/tests/outputs/data_with_search_results_musique_qwen2.5-3b-it-tool-calling0609-think2-all.jsonl",
    # "/mnt/public/kyzhang/MARTI/tests/outputs/data_with_search_results_bamboogle_qwen2.5-3b-it-tool-calling0609-think2-all.jsonl",
    # "/mnt/public/kyzhang/MARTI/data_with_search_results_bamboogle_qwen2.5-3b-it-tool-calling0609-think2-all-t10.jsonl",
    # "/mnt/public/kyzhang/MARTI/data_with_search_results_musique_qwen2.5-3b-it-tool-calling0609-think2-all-t10.jsonl",
    # "/mnt/public/kyzhang/MARTI/data_with_search_results_nq_qwen2.5-3b-it-tool-calling0609-think2-all-t10.jsonl",
    # "musique","bamboogle"
    # "/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/data_with_search_results_bamboogle_qwen2.5-3b-it-async.jsonl",
    # "/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/data_with_search_results_nq_qwen2.5-3b-it-async.jsonl",
    # "/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/data_with_search_results_musique_qwen2.5-3b-it-async.jsonl",
    # "/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/data_with_search_results_bamboogle_qwen2.5-3b-it-async-new.jsonl",
    # "/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/data_with_search_results_nq_qwen2.5-3b-it-async-new.jsonl",
    # "/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/data_with_search_results_musique_qwen2.5-3b-it-async-new.jsonl",
    # new eval
    "/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/outputs/group_norm-tool-search-r1-eval-only/0613/Qwen3-4B/model/eval_results/global_steps1.json",
    "/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/outputs/group_norm-tool-search-r1-eval-only/0613/Qwen3-4B/model/eval_results/global_steps1bamboogle.json",
    "/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/outputs/group_norm-tool-search-r1-eval-only/0613/Qwen3-4B/model/eval_results/global_steps1musique.json",
    "/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/outputs/group_norm-tool-search-r1-eval-only/0613/Qwen3-4B/model/eval_results/global_steps1nq.json",
    # "/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/outputs/group_norm-tool-search-r1-eval-only-wo-think/0613/Qwen3-4B/model/eval_results/global_steps1.json",
    # "/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/outputs/group_norm-tool-search-r1-eval-only-wo-think/0613/Qwen3-4B/model/eval_results/global_steps1bamboogle.json",
    # "/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/outputs/group_norm-tool-search-r1-eval-only-wo-think/0613/Qwen3-4B/model/eval_results/global_steps1musique.json",
    # "/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/outputs/group_norm-tool-search-r1-eval-only-wo-think/0613/Qwen3-4B/model/eval_results/global_steps1nq.json",
    "/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/outputs/group_norm-tool-search-r1-eval-only/0613/Qwen3-8B/model/eval_results/global_steps1.json",
    "/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/outputs/group_norm-tool-search-r1-eval-only/0613/Qwen3-8B/model/eval_results/global_steps1bamboogle.json",
    "/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/outputs/group_norm-tool-search-r1-eval-only/0613/Qwen3-8B/model/eval_results/global_steps1musique.json",
    "/fs-computility/mabasic/qibiqing/marti-project/MARTI-Dev-Git/outputs/group_norm-tool-search-r1-eval-only/0613/Qwen3-8B/model/eval_results/global_steps1nq.json",
]:
    # data = list(srsly.read_jsonl(file))

    # acc_list = []
    # for sample in data:
    #     acc_list.append(compute_score_em(sample["output"], sample["answer"].split("LABEL_SEP")))

    data = srsly.read_json(file)
    acc_list = []
    for sample in data["metadata"][0]:
        acc_list.append(compute_score_em(sample["observation"][-1], sample["label"]))

    file2acc[file] = np.mean(acc_list)
    

pprint(file2acc)

# file2acc = {}
# for file in [
#     "/mnt/public/kyzhang/MARTI/outputs/group_norm-tool-search-r1-tool-call-naive-multi-turn/0609/Qwen2.5-3B-Instruct/model/eval_results/global_steps1.json",
#     "/mnt/public/kyzhang/MARTI/outputs/group_norm-tool-search-r1-tool-call-naive-multi-turn/0609/Qwen2.5-3B-Instruct/model/eval_results/global_steps1bamboogle.json",
#     "/mnt/public/kyzhang/MARTI/outputs/group_norm-tool-search-r1-tool-call-naive-multi-turn/0609/Qwen2.5-3B-Instruct/model/eval_results/global_steps1musique.json",
#     "/mnt/public/kyzhang/MARTI/outputs/group_norm-tool-search-r1-tool-call-naive-multi-turn/0609/Qwen2.5-3B-Instruct/model/eval_results/global_steps1nq.json",
# ]:
#     data = srsly.read_json(file)["metadata"]

#     acc_list = [[]]*len(data[0]["outputs"])
#     print(acc_list)
#     for sample in data:
#         for idx, output in enumerate(sample["outputs"]):
#             acc_list[idx].append(compute_score_em(output, sample["label"].split("LABEL_SEP")))

#     file2acc[file] = np.mean([np.mean(acc) for acc in acc_list])

# print(file2acc)

# Test cases:
# agent_response_example = 'To find the philosophical origins of social work, we need to look at the historical context and the thinking behind the field. Social work emerged as a distinct profession in the late 19th and early 20th centuries, primarily in response to the social and economic inequalities and problems associated with industrialization and urbanization in Europe and the United States.\n\nOne of the key philosophical origins of social work can be traced back to the movements of Social Darwinism marked by William Graham Sumner and Herbert Spencer, who argued for the survival of the fittest and the meritocratic advancement of individuals. However, this view, being somewhat deterministic and individualistic, did not fundamentally address the inequalities and hardships faced by the less fortunate in society.\n\nAnother significant philosophical influence came from the works of Karl Marx and Friedrich Engels, who highlighted the critical role of class struggle and economic inequality in creating social ills. Marx’s theory suggests that the class system in capitalist societies leads to alienation, poverty, and struggle among different social classes. Engels further elucidated the importance of societal structures, gender roles, and economic systems in shaping individual lives, leading to a critique of the existing societal structures and advocating for radical change.\n\nHowever, the more directly influential philosophical ideas for the development of social work can be found in the progressive education movement and the ideas of figures like John Dewey and Jane Addams. John Dewey emphasized the importance of critical thinking, democracy, and the integral connection between individual and society, advocating for participatory democracy and the role of education in promoting social change. Jane Addams, on the other hand, established the settlement house movement, focusing on the role of social workers in addressing local social problems and advocating for community engagement as a means to achieve social reform. Her work emphasized the importance of solidarity and community in solving societal issues.\n\nTherefore, the philosophical origins of social work are deeply rooted in ideas advocating for social justice, equality, and the role of community in addressing societal issues. These ideas reflect a societal critique of the flaws in the capitalist system and the need for collective action to overcome social ills. \n\n<answer> John Dewey and Jane Addams </answer>'
# print(f"Example response score: {check_agent_response_pattern(agent_response_example)}")

# Test with a perfectly structured minimal case:
# perfect_minimal_case = """Initial thought.
# <search>keyword1</search>
# <information>docs1</information>
# Thought before answer.
# <search>keyword1</search>
# <information>docs1</information>
# Thought before answer.
# <search>keyword1</search>
# <information>docs1</information>
# Thought before answer.
# <answer>final answer</answer>"""
# print(f"Perfect minimal case score: {check_agent_response_pattern(perfect_minimal_case)}")

# Test with a perfectly structured minimal case:
# perfect_minimal_case = """Initial thought.
# <search>keyword1</search>
# <information>docs1</information>
# <search>keyword1</search>
# <information>docs1</information>
# Thought before answer.
# <search>keyword1</search>
# <information>docs1</information>
# <answer>final answer</answer>"""
# print(f"Perfect error minimal case score: {check_agent_response_pattern(perfect_minimal_case)}")

# Test with missing thinking process
# missing_tp_case = """Initial thought.
# <search>keyword1</search>
# <information>docs1</information><answer>final answer</answer>""" # Missing TP before answer
# print(f"Missing TP case score: {check_agent_response_pattern(missing_tp_case)}")

# Test with TP between search and info
# tp_between_search_info = """Initial thought.
# <search>keyword1</search>
# Unexpected thought.
# <information>docs1</information>
# Thought before answer.
# <answer>final answer</answer>"""
# print(f"TP between search/info case score: {check_agent_response_pattern(tp_between_search_info)}")

# Test with empty tag content
# empty_tag_content = """Initial thought.
# <search></search>
# <information>docs1</information>
# Thought before answer.
# <answer>final answer</answer>"""
# print(f"Empty tag content case score: {check_agent_response_pattern(empty_tag_content)}")

# Test with no initial thought
# no_initial_tp = """<search>keyword1</search>
# <information>docs1</information>
# Thought before answer.
# <answer>final answer</answer>"""
# print(f"No initial TP case score: {check_agent_response_pattern(no_initial_tp)}")

# Test with trailing text
# trailing_text = """Initial thought.
# <search>keyword1</search>
# <information>docs1</information>
# Thought before answer.
# <answer>final answer</answer>
# Oops, more text here."""
# print(f"Trailing text case score: {check_agent_response_pattern(trailing_text)}")