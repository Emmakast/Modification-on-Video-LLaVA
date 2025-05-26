
import torch
from llava.constants import X_TOKEN_INDEX, DEFAULT_X_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_X_token, get_model_name_from_path, KeywordsStoppingCriteria
import argparse, json, os
from tqdm import tqdm
from datasets import load_dataset

def normalize(text):
    return text.strip().lower()

def inference_single_video(video_path, inp, model, processor):
    print('inference single video for vid:', inp)
    disable_torch_init()
    
    video_processor = processor['video']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    video_tensor = video_processor(video_path, return_tensors='pt')['pixel_values']
    
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)
    key = ['video']
    
    print('tensor shape:', tensor.shape)
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

    
    
    # Loading Video-LLaVA
    model_path = 'LanguageBind/Video-LLaVA-7B'
    device = 'cuda'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)
    
    dataset = load_dataset("morpheushoc/msvd-qa")
    test_set = dataset['test']
    
    predictions = []
    total, correct = 0, 0

    for item in tqdm(test_set):
        video_path = item["video_path"]
        qa_pairs = item["qa"]

        for question_text, gt_answer in qa_pairs:
            # Add prompt if needed
            input_prompt = question_text.strip()

            # Run inference
            pred = inference_single_video(video_path, input_prompt, model, processor)

            # Normalize for accuracy eval
            pred_norm = normalize(pred)
            gt_norm = normalize(gt_answer)

            match = pred_norm == gt_norm
            if match:
                correct += 1
            total += 1

            predictions.append({
                'video_path': video_path,
                'question': question_text,
                'ground_truth': gt_answer,
                'prediction': pred,
                'match': match
            })

    # Save predictions
    os.makedirs(args.output_path, exist_ok=True)
    pred_file = os.path.join(args.output_path, 'msvdqa_predictions.json')
    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=4)

    # Accuracy
    accuracy = correct / total if total > 0 else 0
    print(f"\nEvaluation Complete ✅\nAccuracy: {accuracy:.4f} ({correct}/{total})")