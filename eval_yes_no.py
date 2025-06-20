from prompt_templates import *

import json, os, argparse
from utils.eval_utils import *
from tqdm import tqdm

qtype = "yes_no"
base_prompt = yes_no_evaluation_prompt

def extract_pred(video_llm_output):
    # Extract the yes/no predction from the original video llm output
    video_llm_output = video_llm_output.lower()
    if video_llm_output.startswith("yes"):
        return "yes"
    elif video_llm_output.startswith("no"):
        return "no"
    else:
        return False

def main(predictions, eval_results, output_file, disable_llm):
    for id in tqdm(predictions):

        if id not in eval_results:
            eval_results[id] = {}

        for dim, preds in predictions[id].items():

            if dim in eval_results[id] and eval_results[id][dim] and len(preds)==len(eval_results[id][dim]):    # skip if the eval result already exists
                continue
            eval_results[id][dim] = []

            for pred in preds:
                if "prediction" not in pred and "response" in pred:
                    pred["prediction"] = pred["response"]

                if pred["prediction"] is None:  # In some cases the Video LLM may refuse to produce a response
                    eval_result = {"question": pred["question"], "gt-answer": pred["answer"], "video-llm-prediction": pred["prediction"], "match_success": False, "rating": 0}
                    eval_results[id][dim].append(eval_result)
                    continue
                
                pred["prediction"] = pred["prediction"].replace('</s>', '').strip()
                eval_result = {"question": pred["question"], "gt-answer": pred["answer"], "video-llm-prediction": pred["prediction"], "match_success": True}

                yes_no_pred = extract_pred(pred["prediction"])  # Some hand-crafted matching rules
                if yes_no_pred:
                    eval_result["rating"] = 1 if yes_no_pred==pred["answer"] else 0
                elif disable_llm:
                    eval_result["match_success"] = False    
                    eval_result["rating"] = 0               # Fail to match answer in the video-llm response. Directly set rating to 0
                else:
                    eval_result["match_success"] = False    # Fail to match answer in the video-llm response. Use ChatGPT to evaluate.
                    prompt = f"""{base_prompt}\nYes/No Question:\n{pred["question"]}\nGround-Truth Answer: {pred["answer"]}\nModel Prediction: {pred["prediction"]}"""
                    eval_result["chatgpt-response"], eval_result["rating"] = get_eval_result(prompt)

                eval_results[id][dim].append(eval_result)

    with open(os.path.expanduser(output_file), "w") as f:
        json.dump(eval_results, f, indent=4)

    print_result(eval_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_llm', default="video-llava")
    parser.add_argument('--disable_llm', action='store_true', help="Whether to disable llm evaluation")
    parser.add_argument('--input_path', default="run_video_llava/predictions")
    parser.add_argument('--output_path', default="auto_eval_results")
    parser.add_argument('--number_of_runs', type=int, default=1)
    
    args = parser.parse_args()

    disable_suffix = "_disable_llm" if args.disable_llm else ""

    if args.number_of_runs == 1:
        input_file = f"{args.input_path}/{qtype}.json"
        output_file = f"{args.output_path}{disable_suffix}/{qtype}.json"
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))

        with open(input_file, 'r') as f:
            predictions = json.load(f)

        eval_results = {}

        main(predictions, eval_results, output_file, args.disable_llm)

    else: 
        list_paths = []
        for i in range(args.number_of_runs):
            input_file = f"{args.input_path}/{qtype}_run_{i+1}.json"
            output_file = f"{args.output_path}{disable_suffix}/{qtype}_run_{i+1}.json"
            list_paths.append(input_file)
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))

            with open(input_file, 'r') as f:
                predictions = json.load(f)

            eval_results = {}

            main(predictions, eval_results, output_file, args.disable_llm)

        #print(f'STATISTICS OVER MULTIPLE RUNS for {qtype}')
        #print_result_stats(list_paths)