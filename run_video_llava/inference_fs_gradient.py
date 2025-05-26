import torch
from llava.constants import X_TOKEN_INDEX, DEFAULT_X_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_X_token, get_model_name_from_path, KeywordsStoppingCriteria
import argparse, json, os
from tqdm import tqdm
import pickle
import numpy as np
from torch.nn import CosineSimilarity
from llava.utils import load_frame_embeddings, load_query_embedding, inverse_transform_sampling, select_gradient_boundaries
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ruptures as rpt
from scipy.signal import savgol_filter


answer_prompt = {
    "multi-choice": "\nPlease directly give the best option:",
    "yes_no": "\nPlease answer yes or no:",
    "caption_matching": "\nPlease directly give the best option:",
    "captioning": ""
}

def keypoint_sampling(query_embedding, video_embeddings):
    """
    Samples 8 frame indices from video_embeddings based on cosine similarity to the query_embedding,
    using smoothed signal analysis and breakpoint detection.
    """
    # Step 1: Compute cosine similarity signal
    video_embeddings = np.array(video_embeddings)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    similarity_scores = cosine_similarity(video_embeddings, query_embedding).flatten()

    # Step 2: Smooth the similarity signal
    def smooth_signal(signal, window=11, polyorder=2):
        n = len(signal)
        w = min(window, n)
        w = w if w % 2 == 1 else max(1, w - 1)
        p = min(polyorder, w - 1)
        return savgol_filter(signal, w, p) if n >= w else signal

    smoothed_signal = smooth_signal(similarity_scores)

    # Step 3: Detect breakpoints
    def find_breakpoints(signal, n_bkps=4, min_seg=5):
        if len(signal) < (n_bkps + 1) * min_seg:
            n_bkps = max(0, (len(signal) // min_seg) - 1)
        algo = rpt.Dynp(model="l1", min_size=min_seg).fit(signal)
        bkps = algo.predict(n_bkps=n_bkps)
        return bkps[:-1]  # drop final (len(signal))

    bkps = find_breakpoints(smoothed_signal)

    # Step 4: Find most distinct points around each breakpoint
    def find_max_diff_points(signal, bkps, half_width=5):
        results = []
        for bkpt in bkps:
            start = max(0, bkpt - half_width)
            end = min(len(signal), bkpt + half_width + 1)
            seg = signal[start:end]
            if len(seg) >= 2:
                i_min, i_max = np.argmin(seg), np.argmax(seg)
                idx1 = start + i_min
                idx2 = start + i_max
                if idx1 != idx2:
                    results.extend(sorted([idx1, idx2]))
                else:  # fallback: choose two neighbors
                    if idx1 + 1 < len(signal):
                        results.extend(sorted([idx1, idx1 + 1]))
        return results

    candidate_indices = find_max_diff_points(smoothed_signal, bkps)

    # Step 5: Select top 8 unique indices by their similarity to the query
    unique_indices = list(dict.fromkeys(candidate_indices))  # preserve order
    if len(unique_indices) < 8:
        # Fallback: fill remaining from top global similarities
        sorted_by_similarity = np.argsort(-similarity_scores)
        for idx in sorted_by_similarity:
            if idx not in unique_indices:
                unique_indices.append(idx)
            if len(unique_indices) == 8:
                break
    else:
        unique_indices = unique_indices[:8]

    return sorted(unique_indices)
    
    
def inference_single_video(video_path, inp, model, processor, loaded_query_embeddings_dict, current_task_type):
    print('inference single video for vid:', inp)
    disable_torch_init()
    
    video_processor = processor['video']
    
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    # get precomputed frame (CLIP) embeddings for all frames
    frame_embeddings = load_frame_embeddings(video_embeddings_folder='../video_features/', video_path=video_path)
    total_frames = frame_embeddings.shape[0]

    # get precomputed query embeddings
    current_query_embedding = load_query_embedding(loaded_query_embeddings_dict, inp, current_task_type)

    # NEW: select relevant frames instead of uniform sampling
    selected_frame_indices = select_gradient_boundaries(current_query_embedding, frame_embeddings)
    
    print('selected indices:', selected_frame_indices)

    
    # now we need to pass these indices to the video_processor
    video_tensor = video_processor(video_path, return_tensors='pt', selected_indices=selected_frame_indices)['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)
    key = ['video']
    
    print(f"{roles[1]}: {inp}")
    inp = DEFAULT_X_TOKEN['VIDEO'] + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[tensor, key],
            do_sample=True,
            temperature=0.1,
            max_new_tokens=128,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().replace('</s>', '')
    return outputs

answer_prompt = {
    # "multi-choice": "\nBest Option:",     # The old version
    "multi-choice": "\nPlease directly give the best option:",
    "yes_no": "\nPlease answer yes or no:",
    # "caption_matching": "\nBest Option:",     #The old version
    "caption_matching": "\nPlease directly give the best option:",
    "captioning": ""    # The answer "Generated Caption:" is already contained in the question
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--video_path', default='videos')     
    parser.add_argument('--output_path', default='predictions')     
    parser.add_argument('--task_type', default='multi-choice', choices=['multi-choice', 'captioning', 'caption_matching', 'yes_no'])     
    args = parser.parse_args()

    # Loading questions
    question_path = f"questions/{args.task_type}.json"
    with open(question_path, 'r') as f:
        input_datas = json.load(f)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    pred_file = f"{args.output_path}/{args.task_type}.json"

    predictions = {}
 
    query_dict_pickle_path = '../CLIP_dictionary_all_questions.pkl'
    with open(query_dict_pickle_path, 'rb') as f:
        loaded_query_embeddings = pickle.load(f)
    print(f"Successfully loaded query embeddings from {query_dict_pickle_path}")

    # Loading Video-LLaVA
    model_path = 'LanguageBind/Video-LLaVA-7B'
    device = 'cuda'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)
    
    # for vid, data in tqdm(input_datas.items()):
    #     if vid not in predictions:
    #         predictions[vid] = {}
    #         video_path = os.path.join(args.video_path, f'{vid}.mp4')
    #         for dim, questions in data.items():
    #             predictions[vid][dim] = []
    #             for question in questions:
    #                 inp = question['question'] + answer_prompt[args.task_type]
    #                 video_llm_pred = inference_single_video(video_path, inp, model, processor, loaded_query_embeddings, args.task_type)
    #                 predictions[vid][dim].append({'question': question['question'], 'answer': question['answer'], 'prediction': video_llm_pred})
    #         with open(pred_file, 'w') as f:
    #             json.dump(predictions, f, indent=4)

    for i in range(3):
        # Create a unique prediction file name for each run
        pred_file = f"{args.output_path}/{args.task_type}_run_{i+1}.json"

        # Initialize predictions for each run
        predictions = {}

        for vid, data in tqdm(input_datas.items()):
            if vid not in predictions:
                predictions[vid] = {}
                video_path = os.path.join(args.video_path, f'{vid}.mp4')
                for dim, questions in data.items():
                    predictions[vid][dim] = []
                    for question in questions:
                        inp = question['question'] + answer_prompt[args.task_type]
                        video_llm_pred = inference_single_video(video_path, inp, model, processor, loaded_query_embeddings, args.task_type)
                        predictions[vid][dim].append({'question': question['question'], 'answer': question['answer'], 'prediction': video_llm_pred})

        # Save predictions for the current run after processing all videos
        with open(pred_file, 'w') as f:
            json.dump(predictions, f, indent=4)