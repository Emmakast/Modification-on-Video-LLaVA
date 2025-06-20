import json, random, time, requests, re
import numpy as np

url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer $OPENAI_API_KEY"
}

def print_result(eval_results):
    with open("meta_info.json", 'r') as f:
        meta_infos = json.load(f)
    match_rate = 0  # the success rate of rule-based answer matching
    result_asp = {'action': 0, 'direction': 0, 'speed': 0, 'order': 0, 'attribute_change': 0}   # eval result under every temporal aspect
    qcount_asp = {'action': 0, 'direction': 0, 'speed': 0, 'order': 0, 'attribute_change': 0}   # question count result under every temporal aspect
    result_fasp = {'fine-grained action': 0, 'coarse-grained action': 0, 'object motion': 0, 'camera motion': 0, 
                   'absolute speed': 0, 'relative speed': 0, 'order': 0, 'color & light change': 0, 'size & shape change': 0, 'combined change': 0, 'other change': 0}  # eval result under every fine-grained temporal aspect
    qcount_fasp = {'fine-grained action': 0, 'coarse-grained action': 0, 'object motion': 0, 'camera motion': 0, 
                   'absolute speed': 0, 'relative speed': 0, 'order': 0, 'color & light change': 0, 'size & shape change': 0, 'combined change': 0, 'other change': 0}  # question count result under every fine-grained temporal aspect
    
    for id in eval_results:
        for asp in eval_results[id]:
            fasp = meta_infos[id.replace('.jpg', '').replace('.mp4', '')]["eval_dim"][asp]["type"] if asp!="order" else "order"
            for result in eval_results[id][asp]:
                result_asp[asp] += result["rating"]
                result_fasp[fasp] += result["rating"]
                qcount_asp[asp] += 1
                qcount_fasp[fasp] += 1
                if "match_success" in result:
                    match_rate += result["match_success"]

    match_rate = round(match_rate/sum(qcount_asp.values())*100, 1)
    result_asp['avg'] = round(sum(result_asp.values())*100/sum(qcount_asp.values()), 1)
    for asp in result_asp:
        if asp!='avg':
            result_asp[asp] = round(result_asp[asp]*100/qcount_asp[asp], 1)
    for fasp in result_fasp:
        result_fasp[fasp] = round(result_fasp[fasp]*100/qcount_fasp[fasp], 1)
    print("Accuracy Results:")
    print(result_asp)
    print(result_fasp)
    print(f"Match Success Rate={match_rate}")

def parse_llm_output(llm_output, qtype):
    datas = []
    if qtype=='multi-choice':
        print(llm_output)
        llm_output = llm_output.replace("Multi-Choice Question:", "")
        llm_output = llm_output.replace("Multi-Choice Question", "")
        if "[SEP]" in llm_output:
            qa_pairs = llm_output.split("[SEP]")
        else:
            qa_pairs = llm_output.split("\n\n")
        for qa_pair in qa_pairs:
            # Skip empty strings
            if qa_pair.strip():
                if qa_pair.count("Correct Answer:")!=1:
                    return None
                # Split the pair into question and answer parts
                question, answer = qa_pair.strip().split("Correct Answer:")
                question_text = question.strip()
                answer_text = answer.strip()
                datas.append({"question": question_text, "answer": answer_text})

    elif qtype=='yes_no':
        # Split the text into lines
        lines = llm_output.split("\n")

        # Iterate through each line
        for line in lines:
            line = line.strip()
            # Check if the line indicates positive or negative questions
            if "Positive Questions" in line or "Negative Questions" in line:
                continue
            # Parse the JSON data and extract the question
            if line and "question" in line and "answer" in line:
                datas.append(json.loads(line))

    elif qtype=='caption_matching':
        question_templates = [
            "Which caption matches the video better?\nCaption A: [caption_a]\nCaption B: [caption_b]",
            "Which description is a more suitable match for the video?\nOption 1: [caption_a]\nOption 2: [caption_b]",
            "Which sentence better captures the essence of the video?\nSentence A: [caption_a]\nSentence B: [caption_b]"
        ]
        # Define regular expressions for true and false captions
        false_caption_pattern = re.compile(r'False Captions:\s*(.*)', re.DOTALL)

        # Extract true caption
        true_caption = llm_output.split("False Captions")[0].replace("True Caption:", "").strip()

        # Extract false captions into a list
        false_captions_match = false_caption_pattern.search(llm_output)
        false_captions_text = false_captions_match.group(1).strip() if false_captions_match else ""
        false_captions_list = [caption.strip() for caption in false_captions_text.split('\n') if caption.strip()]
        random.shuffle(question_templates)
        for fal_cap, template in zip(false_captions_list, question_templates):
            answer_index = random.choice([1,2])
            if answer_index==1:
                question = template.replace("[caption_a]", true_caption).replace("[caption_b]", fal_cap)
            elif answer_index==2:
                question = template.replace("[caption_a]", fal_cap).replace("[caption_b]", true_caption)
            answer = question.split("\n")[answer_index]
            data = {"question": question, "answer": answer}
            datas.append(data)
    return datas

def llm_output_to_rating(llm_output):
    assert 'Correct' in llm_output or 'Incorrect' in llm_output
    if llm_output.startswith('Correct'):
        rating = 1
    elif llm_output.startswith('Incorrect'):
        rating = 0
    elif ('Correct' in llm_output) and ('Incorrect' not in llm_output):
        rating = 1
    elif 'Incorrect' in llm_output:
        rating = 0
    return rating

def get_llm_output(prompt, sys_prompt, max_tokens=128):
    if sys_prompt is None:
        sys_prompt = "You are an AI assistant for question answering."
    data = {
        "max_tokens": max_tokens,
        "model": "gpt-3.5-turbo-1106",
        "temperature": 1.0,
        "top_p": 1,
        "presence_penalty": 1,
        "messages": [
            {
                "role": "system",
                "content": sys_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data).encode('utf-8'))
    result = response.content.decode("utf-8")
    dict_result = json.loads(result)
    llm_output = dict_result['choices'][0]['message']['content'].strip()
    return llm_output

def get_gen_question(prompt, maxtry=10, sys_prompt=None, qtype='multi-choice', max_tokens=1000):
    llm_output, extracted_questions = None, None
    while True:
        try:
            llm_output = get_llm_output(prompt, sys_prompt, max_tokens)
            extracted_questions = parse_llm_output(llm_output, qtype)
            return extracted_questions
        except:
            if maxtry<=0:
                return extracted_questions
            maxtry -= 1
            print(f"Not success! {maxtry} retries remaining...")
            time.sleep(random.uniform(1, 2))

def get_eval_result(prompt, maxtry=10, sys_prompt=None):
    llm_output = None
    while True:
        try:
            llm_output = get_llm_output(prompt, sys_prompt)
            rating = llm_output_to_rating(llm_output)
            return llm_output, rating
        except:
            if maxtry<=0:
                return llm_output, 0
            maxtry -= 1
            print(f"Not success! {maxtry} retries remaining...")
            time.sleep(random.uniform(1, 2))

def process_gemini_caption(llm_output):
    caption = llm_output.strip().split('\n')[-1].replace('*', '').strip().strip('"')
    if ':' in caption:
        caption = caption.split(':')[1].strip().strip('"')
    return caption

def process_reka_caption(llm_output):
    last_line = llm_output.split('\n')[-1]
    if llm_output.count("\"")==2:
        matches = re.findall(r'"(.*?)"', llm_output)
        caption = matches[0]
    elif last_line.count("\"")==2:
        matches = re.findall(r'"(.*?)"', last_line)
        caption = matches[0]
    elif llm_output.count(": ")==1:
        caption = llm_output.split(": ")[1]
    elif last_line.count(": ")==1:
        caption = last_line.split(": ")[1]
    else:
        caption = llm_output
    return caption

def _compute_single(eval_results):
    """Compute metrics for one run; returns (result_asp, result_fasp, match_rate)."""
    with open("meta_info.json", 'r') as f:
        meta_infos = json.load(f)

    # initialize
    match_count = 0
    result_asp = {asp: 0 for asp in ['action','direction','speed','order','attribute_change']}
    qcount_asp = result_asp.copy()
    fasp_keys = [
        'fine-grained action','coarse-grained action','object motion','camera motion',
        'absolute speed','relative speed','order','color & light change',
        'size & shape change','combined change','other change'
    ]
    result_fasp = {k: 0 for k in fasp_keys}
    qcount_fasp = result_fasp.copy()

    # accumulate
    for vid_id, asp_dict in eval_results.items():
        for asp, results in asp_dict.items():
            # determine fine‐grained key
            if asp == "order":
                fasp = "order"
            else:
                key = vid_id.rsplit('.',1)[0]
                fasp = meta_infos[key]["eval_dim"][asp]["type"]
            for r in results:
                rating = r["rating"]
                result_asp[asp] += rating
                result_fasp[fasp] += rating
                qcount_asp[asp] += 1
                qcount_fasp[fasp] += 1
                if r.get("match_success") is not None:
                    match_count += r["match_success"]

    # finalize percentages
    total_q = sum(qcount_asp.values())
    match_rate = match_count / total_q * 100

    # accuracy per asp
    for asp in result_asp:
        result_asp[asp] = result_asp[asp] / qcount_asp[asp] * 100
    result_asp['avg'] = sum(result_asp.values()) / len(result_asp)

    # accuracy per fine‐grained asp
    for fasp in result_fasp:
        result_fasp[fasp] = result_fasp[fasp] / qcount_fasp[fasp] * 100

    return result_asp, result_fasp, match_rate

def print_result_stats(json_paths):
    """
    json_paths: list of file paths to eval_results JSON files (one per run)
    Loads each JSON, computes mean and std across runs, and prints them.
    """
    # Load each run from its JSON file
    runs = []
    for path in json_paths:
        with open(path, 'r') as f:
            runs.append(json.load(f))

    # containers: {metric: [val_run1, val_run2, ...]}
    asp_metrics = {}
    fasp_metrics = {}
    match_rates = []

    # collect metrics from each run
    for run in runs:
        r_asp, r_fasp, mr = _compute_single(run)
        match_rates.append(mr)
        for k, v in r_asp.items():
            asp_metrics.setdefault(k, []).append(v)
        for k, v in r_fasp.items():
            fasp_metrics.setdefault(k, []).append(v)

    # print header
    print("=== Accuracy over Aspects (mean ± std) ===")
    for asp, vals in asp_metrics.items():
        mu, sd = np.mean(vals), np.std(vals, ddof=1)
        print(f"{asp:20s}: {mu:6.2f}% ± {sd:5.2f}")

    print("\n=== Accuracy over Fine-grained Aspects ===")
    for fasp, vals in fasp_metrics.items():
        mu, sd = np.mean(vals), np.std(vals, ddof=1)
        print(f"{fasp:20s}: {mu:6.2f}% ± {sd:5.2f}")

    mu_m, sd_m = np.mean(match_rates), np.std(match_rates, ddof=1)
    print(f"\nMatch Success Rate : {mu_m:6.2f}% ± {sd_m:5.2f}")