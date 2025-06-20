import torch
from llava.constants import X_TOKEN_INDEX, DEFAULT_X_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_X_token, get_model_name_from_path, KeywordsStoppingCriteria
import argparse, json, os
from tqdm import tqdm

def inference_single_video(video_path, inp, model, processor):
    print('inference single video for vid:', inp)
    disable_torch_init()
    
    video_processor = processor['video']
    conv_mode = "llava_v2"
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', default='videos')
    parser.add_argument('--output_path', default='predictions_prompt')
    parser.add_argument('--task_type', default='multi-choice', choices=['multi-choice', 'captioning', 'caption_matching', 'yes_no'])
    paser.add_argument('--answer_prompt', default="\nApproach the video by thinking about the reasons behind the actions and their order in time, and choose the most relevant option.")
    args = parser.parse_args()

    # Loading questions
    question_path = f"questions/{args.task_type}.json"
    with open(question_path, 'r') as f:
        input_datas = json.load(f)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Loading Video-LLaVA
    model_path = 'LanguageBind/Video-LLaVA-7B'
    device = 'cuda'
    load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)

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
                        inp = question['question'] + args.answer_prompt
                        video_llm_pred = inference_single_video(video_path, inp, model, processor)
                        predictions[vid][dim].append({'question': question['question'], 'answer': question['answer'], 'prediction': video_llm_pred})

        # Save predictions for the current run after processing all videos
        with open(pred_file, 'w') as f:
            json.dump(predictions, f, indent=4)
