import torch
from llava.constants import X_TOKEN_INDEX, DEFAULT_X_TOKEN, IMAGE_TOKEN_INDEX # IMAGE_TOKEN_INDEX is the target, X_TOKEN_INDEX['VIDEO'] might be what tokenizer_X_token uses as placeholder key
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_X_token, get_model_name_from_path, KeywordsStoppingCriteria
import argparse, json, os
from tqdm import tqdm
import traceback

# Standard LLaVA value for the special image/video token placeholder in input_ids
# The tokenizer_X_token should replace the placeholder string with this IMAGE_TOKEN_INDEX.
# If it's inserting X_TOKEN_INDEX['VIDEO'] directly and that's different, it's an issue.
# For now, we assume tokenizer_X_token correctly produces IMAGE_TOKEN_INDEX (-200).
# The warning about -201 suggests that X_TOKEN_INDEX['VIDEO'] might be -201.
# We will search for IMAGE_TOKEN_INDEX in input_ids. If tokenizer_X_token uses
# X_TOKEN_INDEX['VIDEO'] as the *value* it inserts, and that value is -201,
# then the search for -200 will fail. This needs to be consistent.
# Let's assume tokenizer_X_token correctly inserts IMAGE_TOKEN_INDEX = -200.

# >>> START OF M1 RELATED PLACEHOLDER <<<
def get_m1_temporal_weights(video_frames_tensor_for_m1, query_text, m1_model, m1_clip_processor, m1_clip_model, device):
    print(f"[M1 Placeholder] M1 processing query: '{query_text[:50]}...'")
    num_frames = video_frames_tensor_for_m1.shape[0] 
    print(f"[M1 Placeholder] Number of frames for M1: {num_frames}, expected input shape [T,C,H,W], actual: {video_frames_tensor_for_m1.shape}")
    dummy_weights = torch.softmax(torch.randn(num_frames, device=device), dim=-1)
    print(f"[M1 Placeholder] Dummy weights: {dummy_weights.tolist()}")
    return dummy_weights.to(dtype=torch.float16)
# >>> END OF M1 RELATED PLACEHOLDER <<<

def inference_single_video(video_path, inp, model, processor, tokenizer, 
                           m1_model_placeholder, m1_clip_processor_placeholder, m1_clip_model_placeholder):
    print(f"\n--- Entering inference_single_video for video: {video_path} ---")
    disable_torch_init()
    
    video_processor = processor['video']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    
    try:
        video_tensor_from_processor_raw = video_processor(video_path, return_tensors='pt')['pixel_values']
    except Exception as e:
        print(f"Error during video_processor call for {video_path}: {e}")
        return f"[Error: Video processing exception: {e}]"

    if isinstance(video_tensor_from_processor_raw, list):
        print("Warning: video_tensor_from_processor_raw is a list. Using the first element.")
        if not video_tensor_from_processor_raw: return "[Error: Empty video_tensor_from_processor_raw list]"
        llava_input_tensor = video_tensor_from_processor_raw[0].to(model.device, dtype=torch.float16)
        if llava_input_tensor.ndim == 4: llava_input_tensor = llava_input_tensor.unsqueeze(0) 
    else:
        llava_input_tensor = video_tensor_from_processor_raw.to(model.device, dtype=torch.float16)

    if not (llava_input_tensor.ndim == 5 and llava_input_tensor.shape[0] == 1):
        print(f"Error: Unexpected LLaVA input tensor shape: {llava_input_tensor.shape}. Expected (1, C, T, H, W).")
        return "[Error: LLaVA input tensor shape mismatch]"
    
    print('LLaVA input tensor shape (B,C,T,H,W):', llava_input_tensor.shape)
    
    m1_status_message = "M1_Not_Run"; temporal_weights = None; projected_weighted_features = None 
    generation_path_status = "Attempting_M1_Path"; outputs_text = "[Default output text if no path completes]" 

    try:
        frames_for_m1_input = llava_input_tensor.squeeze(0).permute(1, 0, 2, 3)
        temporal_weights = get_m1_temporal_weights(frames_for_m1_input, inp, m1_model_placeholder, m1_clip_processor_placeholder, m1_clip_model_placeholder, model.device)
        print('temporal weights shape:', temporal_weights.shape)

        batch_size, C_dim, T_dim, H_dim, W_dim = llava_input_tensor.shape
        frames_permuted_for_llava_tower = llava_input_tensor.permute(0, 2, 1, 3, 4) 
        frames_reshaped_for_llava_tower = frames_permuted_for_llava_tower.reshape(batch_size * T_dim, C_dim, H_dim, W_dim)
        
        vision_tower = model.get_model().get_video_tower() 
        vision_tower_output = vision_tower(frames_reshaped_for_llava_tower)
        raw_frame_features = vision_tower_output
        if isinstance(raw_frame_features, tuple): raw_frame_features = raw_frame_features[0]
        print(f"Raw frame features from vision tower, shape: {raw_frame_features.shape}")
        if raw_frame_features.ndim == 3 and raw_frame_features.shape[1] > 1 : 
            raw_frame_features = raw_frame_features[:, 0]
            print(f"Features after CLS token selection, shape: {raw_frame_features.shape}")
        
        feature_dim = raw_frame_features.shape[-1]
        raw_frame_features_reshaped = raw_frame_features.view(batch_size, T_dim, feature_dim)
        if temporal_weights.ndim == 1: temporal_weights_expanded = temporal_weights.unsqueeze(0).unsqueeze(-1)
        elif temporal_weights.ndim == 2 and temporal_weights.shape[0] == batch_size: temporal_weights_expanded = temporal_weights.unsqueeze(-1) 
        else: raise ValueError(f"temporal_weights shape {temporal_weights.shape} not compatible")
        
        weighted_frame_features_intermediate = raw_frame_features_reshaped * temporal_weights_expanded
        weighted_frame_features_for_projector = weighted_frame_features_intermediate.reshape(batch_size * T_dim, feature_dim)
        print(f"Weighted features for projector, shape: {weighted_frame_features_for_projector.shape}")
        mm_projector = model.get_model().mm_projector
        projected_weighted_features = mm_projector(weighted_frame_features_for_projector) 
        print(f"Projected weighted features, shape: {projected_weighted_features.shape}")
        m1_status_message = f"M1_Features_Weighted_Projected(shape:{projected_weighted_features.shape})"
    except Exception as e:
        print(f"Error during M1/Feature Extraction/Re-weighting path: {e}"); traceback.print_exc()
        m1_status_message = f"M1_Error_{type(e).__name__}"; projected_weighted_features = None 
        generation_path_status = "Fallback_Due_To_M1_Error"

    llava_prompt_inp = DEFAULT_X_TOKEN['VIDEO'] + '\n' + inp 
    conv.append_message(conv.roles[0], llava_prompt_inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    try:
        # It's crucial that tokenizer_X_token replaces DEFAULT_X_TOKEN['VIDEO'] with IMAGE_TOKEN_INDEX (-200)
        input_ids_cpu = tokenizer_X_token(prompt, tokenizer, DEFAULT_X_TOKEN['VIDEO'], IMAGE_TOKEN_INDEX, return_tensors='pt')
        # The above line was previously: tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], ...)
        # The third argument should be the *string* to replace, the fourth should be the *ID to replace it with*.
        # X_TOKEN_INDEX['VIDEO'] might be -201, which caused the warning if it was used as replacement ID.
        # We want IMAGE_TOKEN_INDEX (-200) to be in input_ids.

        vocab_size = tokenizer.vocab_size
        valid_mask = ((input_ids_cpu >= 0) & (input_ids_cpu < vocab_size)) | (input_ids_cpu == IMAGE_TOKEN_INDEX)
        if (~valid_mask).any():
            print(f"Warning: Invalid token IDs found in input_ids_cpu: {input_ids_cpu[~valid_mask].tolist()}")
        input_ids = input_ids_cpu.unsqueeze(0).cuda()
    except Exception as e:
         print(f"Error during tokenizer_X_token or input_ids inspection: {e}"); traceback.print_exc()
         outputs_text = f"[Error during tokenization: {e}]"
         if temporal_weights is not None: return f"[{m1_status_message}; GenPath:Tokenization_Error; M1W:{temporal_weights.tolist()}] {outputs_text}"
         else: return f"[{m1_status_message}; GenPath:Tokenization_Error] {outputs_text}"

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    if projected_weighted_features is not None and generation_path_status == "Attempting_M1_Path": # Ensure M1 path didn't error out
        try:
            print("Attempting generation with re-weighted features...")
            num_video_tokens_per_batch_item = projected_weighted_features.shape[0] // batch_size 
            video_feature_embeddings = projected_weighted_features.view(batch_size, num_video_tokens_per_batch_item, -1)
            current_input_ids_unbatched = input_ids[0] 

            video_token_indices = torch.where(current_input_ids_unbatched == IMAGE_TOKEN_INDEX)[0]
            print(f"Searching for IMAGE_TOKEN_INDEX ({IMAGE_TOKEN_INDEX}), Found at indices: {video_token_indices.tolist()}")
            
            if len(video_token_indices) == 0:
                print("CRITICAL Warning: IMAGE_TOKEN_INDEX (-200) not found in input_ids. Cannot inject video features for M1 path.")
                generation_path_status = "Fallback_No_Video_Token_Found"
            else: 
                video_token_pos = video_token_indices[0].item() 
                new_inputs_embeds_parts = []
                
                ids_before_video = current_input_ids_unbatched[:video_token_pos]
                if ids_before_video.numel() > 0:
                    if (ids_before_video == IMAGE_TOKEN_INDEX).any(): raise ValueError("IMAGE_TOKEN_INDEX in ids_before_video")
                    embeds_before_video = model.get_model().embed_tokens(ids_before_video.unsqueeze(0))[0]
                else: embeds_before_video = torch.empty((0, model.config.hidden_size), dtype=video_feature_embeddings.dtype, device=model.device)
                new_inputs_embeds_parts.append(embeds_before_video)
                
                new_inputs_embeds_parts.append(video_feature_embeddings[0]) 
                
                ids_after_video = current_input_ids_unbatched[video_token_pos + 1:]
                if ids_after_video.numel() > 0:
                    if (ids_after_video == IMAGE_TOKEN_INDEX).any(): raise ValueError("IMAGE_TOKEN_INDEX in ids_after_video")
                    embeds_after_video = model.get_model().embed_tokens(ids_after_video.unsqueeze(0))[0]
                else: embeds_after_video = torch.empty((0, model.config.hidden_size), dtype=video_feature_embeddings.dtype, device=model.device)
                new_inputs_embeds_parts.append(embeds_after_video)
                
                final_inputs_embeds = torch.cat(new_inputs_embeds_parts, dim=0).unsqueeze(0) 
                print(f"Constructed final_inputs_embeds shape: {final_inputs_embeds.shape}")

                original_attention_mask = model.prepare_attention_mask_for_generation(input_ids, tokenizer.pad_token_id, tokenizer.eos_token_id)[0]
                mask_part1 = original_attention_mask[:video_token_pos]
                mask_part2 = torch.ones(num_video_tokens_per_batch_item, dtype=torch.long, device=model.device) 
                mask_part3 = original_attention_mask[video_token_pos + 1:]
                final_attention_mask = torch.cat([mask_part1, mask_part2, mask_part3], dim=0).unsqueeze(0) 
                print(f"Constructed final_attention_mask shape: {final_attention_mask.shape}")

            if not generation_path_status.startswith("Fallback"):
                with torch.inference_mode():
                    output_ids = model.generate(
                        inputs_embeds=final_inputs_embeds, attention_mask=final_attention_mask, 
                        do_sample=True, temperature=0.1, max_new_tokens=128, use_cache=True,
                        stopping_criteria=[KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] 
                    )
                outputs_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().replace('</s>', '')
                generation_path_status = "Success_M1_Path"
        except Exception as e:
            print(f"Error during generation with re-weighted features: {e}"); traceback.print_exc()
            outputs_text = "[Error in M1 generation path]"; generation_path_status = "Fallback_Due_To_M1_Generate_Error" 

    if generation_path_status.startswith("Fallback"):
        print(f"--- ({generation_path_status}) Calling original model.generate() for video: {video_path} ---")
        # Use llava_input_tensor for the fallback, as it's the processed tensor for LLaVA
        llava_generate_images_arg = [llava_input_tensor, ['video']]
        with torch.inference_mode():
            output_ids_fallback = model.generate(
                input_ids, images=llava_generate_images_arg,
                do_sample=True, temperature=0.1, max_new_tokens=128, use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs_text = tokenizer.decode(output_ids_fallback[0, input_ids.shape[1]:]).strip().replace('</s>', '')

    if temporal_weights is not None: outputs = f"[{m1_status_message}; GenPath:{generation_path_status}; M1W:{temporal_weights.tolist()}] {outputs_text}"
    else: outputs = f"[{m1_status_message}; GenPath:{generation_path_status}] {outputs_text}"
    print(f"--- Exiting inference_single_video. Output snippet: {outputs[:120]}... ---")
    return outputs

answer_prompt = {
    "multi-choice": "\nPlease directly give the best option:",
    "yes_no": "\nPlease answer yes or no:",
    "caption_matching": "\nPlease directly give the best option:",
    "captioning": "" 
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--video_path', default='videos')     
    parser.add_argument('--output_path', default='predictions')     
    parser.add_argument('--task_type', default='multi-choice', choices=['multi-choice', 'captioning', 'caption_matching', 'yes_no'])     
    args = parser.parse_args()

    m1_model_instance = None; m1_clip_processor_instance = None; m1_clip_model_instance = None
    question_path = f"questions/{args.task_type}.json"
    with open(question_path, 'r') as f: input_datas = json.load(f)
    task_output_path = os.path.join(args.output_path, args.task_type)
    if not os.path.exists(task_output_path): os.makedirs(task_output_path)
    pred_file = os.path.join(task_output_path, f"{args.task_type}.json")
    predictions = {}
    model_path = 'LanguageBind/Video-LLaVA-7B' 
    device = 'cuda'; load_4bit, load_8bit = True, False
    model_name = get_model_name_from_path(model_path)
    print("Loading Video-LLaVA model..."); tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)
    print('Video-LLaVA model loaded.')
    counter = 0
    print('Looping over input data items....')
    for vid, data in tqdm(input_datas.items()):
        if counter == 1: break
        if vid not in predictions:
            predictions[vid] = {}
            current_video_file_path = os.path.join(args.video_path, f'{vid}.mp4')
            if not os.path.exists(current_video_file_path):
                print(f"Warning: Video file not found {current_video_file_path}, skipping.")
                predictions[vid]['error'] = 'Video file not found'; continue
            for dim, questions in data.items():
                predictions[vid][dim] = []
                for question in questions:
                    inp_text = question['question'] + answer_prompt[args.task_type]
                    video_llm_pred = inference_single_video(current_video_file_path, inp_text, model, processor, tokenizer, m1_model_instance, m1_clip_processor_instance, m1_clip_model_instance)
                    predictions[vid][dim].append({'question': question['question'], 'answer': question['answer'], 'prediction': video_llm_pred})
            with open(pred_file, 'w') as f: json.dump(predictions, f, indent=4)
        counter += 1
    print(f"Processing complete. Predictions saved to {pred_file}")
