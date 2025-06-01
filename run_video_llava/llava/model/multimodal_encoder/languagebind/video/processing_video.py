import torch
import cv2
import decord
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from torchvision import transforms
from transformers import ProcessorMixin, BatchEncoding
from transformers.image_processing_utils import BatchFeature
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms._transforms_video import NormalizeVideo, RandomCropVideo, RandomHorizontalFlipVideo, CenterCropVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample

import os

decord.bridge.set_bridge('torch')

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

answer_prompt = {
    "multi-choice": "\nPlease directly give the best option:",
    "yes_no": "\nPlease answer yes or no:",
    "caption_matching": "\nPlease directly give the best option:",
    "captioning": ""
}

def make_list_of_images(x):
    if not isinstance(x, list):
        return [x]
    return x

def get_video_transform(config):
    config = config.vision_config
    if config.video_decode_backend == 'pytorchvideo':
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(config.num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                    ShortSideScale(size=224),
                    CenterCropVideo(224),
                    RandomHorizontalFlipVideo(p=0.5),
                ]
            ),
        )

    elif config.video_decode_backend == 'decord':

        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                RandomHorizontalFlipVideo(p=0.5),
            ]
        )

    elif config.video_decode_backend == 'opencv':
        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                RandomHorizontalFlipVideo(p=0.5),
            ]
        )
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return transform

def get_next_frame_sample(sampled_idx, cv2_vr, total_frames):
    frame1_cv_idx = -1
    frame2_cv_idx = -1

    if sampled_idx == total_frames - 1:
        if sampled_idx > 0: 
            frame1_cv_idx = sampled_idx - 1
            frame2_cv_idx = sampled_idx
    else: 
        frame1_cv_idx = sampled_idx
        frame2_cv_idx = sampled_idx + 1
        
    cv2_vr.set(cv2.CAP_PROP_POS_FRAMES, frame1_cv_idx)
    _, frame1_bgr = cv2_vr.read()
        
    cv2_vr.set(cv2.CAP_PROP_POS_FRAMES, frame2_cv_idx)
    _, frame2_bgr = cv2_vr.read()

    frame1_rgb = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2RGB)
    frame1_tensor = torch.from_numpy(frame1_rgb).permute(2, 0, 1)

    frame2_rgb = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2RGB)
    frame2_tensor = torch.from_numpy(frame2_rgb).permute(2, 0, 1)

    return frame1_tensor, frame2_tensor

def get_frame_pairs_for_flow(video_path, num_sampled_frames, sampled_idx=None):
    """
    Fetches pairs of frames for optical flow.
    For a sampled frame t:
    - If t is the last frame of the video, use (original_video_frame_t-1, original_video_frame_t).
    - Otherwise, use (original_video_frame_t, original_video_frame_t+1).
    Returns a list of tuples, where each tuple contains two PyTorch tensors (RGB, C, H, W)
    representing (frame1_for_flow_calc, frame2_for_flow_calc).
    """
    cv2_vr = cv2.VideoCapture(video_path)
    total_frames = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if not sampled_idx:
        sampled_frame_id_list = np.linspace(0, total_frames - 1, num_sampled_frames, dtype=int)
    else:
        sampled_frame_id_list = [sampled_idx]
    
    frame_pairs = [] # Stores (tensor_frame1, tensor_frame2)
    for sampled_idx in sampled_frame_id_list:
        frame1_tensor, frame2_tensor = get_next_frame_sample(sampled_idx, cv2_vr, total_frames)
        frame_pairs.append((frame1_tensor, frame2_tensor))

    cv2_vr.release()
    return frame_pairs

def compute_optical_flow(tensor_frame_1, tensor_frame_2):
    frame1_np = tensor_frame_1.permute(1, 2, 0).numpy()
    frame2_np = tensor_frame_2.permute(1, 2, 0).numpy()

    if frame1_np.dtype != np.uint8:
        frame1_np = frame1_np.astype(np.uint8)
    if frame2_np.dtype != np.uint8:
        frame2_np = frame2_np.astype(np.uint8)

    gray1 = cv2.cvtColor(frame1_np, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2_np, cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev=gray1,
        next=gray2,
        flow=None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.1, flags=0
    )
    return flow

def visualize_flow_arrows_on_image(base_frame, flow_field, step=16, scale=1.0, tip_length=0.4, epsilon=1):
    """
    Visualizes optical flow using arrows on the base image.
    Args:
        base_frame: The first frame of the pair
        flow_field (numpy.ndarray): A (height, width, 2) array with dx, dy components.
        step (int): Draw an arrow every 'step' pixels.
        scale (float): Multiplier for arrow length for better visibility.
    Returns:
        numpy.ndarray: An RGB image (H, W, C) with flow arrows drawn on it.
    """
    # Convert base frame tensor (C, H, W, RGB) to NumPy array (H, W, C, RGB)
    if base_frame.dtype != np.uint8:
        base_frame = base_frame.astype(np.uint8)

    # Make a copy to draw on
    vis_image = np.copy(base_frame)
    h, w = vis_image.shape[:2]

    # Iterate over a sparse grid
    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = flow_field[y, x]

            start_point = (int(x), int(y))
            end_point = (int(x + dx * scale), int(y + dy * scale))

            if np.abs(start_point[0] - end_point[0]) > epsilon or np.abs(start_point[1] - end_point[1]) > epsilon:
                cv2.arrowedLine(vis_image, start_point, end_point, (0, 255, 0), 3, tipLength=tip_length)
    
    return vis_image

def video_frame_decoration(frame, time, mode, flow_field=None):
    if mode == "optical_flow_arrow":

        img_with_arrows = visualize_flow_arrows_on_image(frame, 
                                                        flow_field, 
                                                        step=40,
                                                        scale=3.0,
                                                        tip_length=0.3,
                                                        epsilon=25)
        
        return img_with_arrows
        
    elif mode is None:
        return frame
    else:
        raise NotImplementedError

def load_and_transform_video(
        video_path,
        transform,
        video_decode_backend='opencv',
        clip_start_sec=0.0,
        clip_end_sec=None,
        num_frames=8,
        selected_indices=None # DANIEL: ADDED for dynamic frame selection
):
    if video_decode_backend == 'pytorchvideo':
        #  decord pyav
        video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
        duration = video.duration
        start_sec = clip_start_sec  # secs
        end_sec = clip_end_sec if clip_end_sec is not None else duration  # secs
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_outputs = transform(video_data)

    elif video_decode_backend == 'decord':
        decord.bridge.set_bridge('torch')
        decord_vr = VideoReader(video_path, ctx=cpu(0))
        duration = len(decord_vr)
        
        # DANIEL: added
        if selected_indices is not None:
            frame_id_list = selected_indices
            
        else:
            frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)

        # JUTTE: added storing the time in seconds of each frame
        fps = decord_vr.get_avg_fps() 
        time_list = [idx / fps for idx in frame_id_list]
        np.save('/home/scur0554/TempCompass/run_video_llava/time_list.npy', time_list)

        video_data = decord_vr.get_batch(frame_id_list)
        decorated = []
        mode = None # "optical_flow_arrow"

        if mode:
            # Find flow fields for the images in video_data
            frame_pairs_for_flow = get_frame_pairs_for_flow(video_path, num_frames)
            all_flow_fields = []
            for i, (frame1_tensor, frame2_tensor) in enumerate(frame_pairs_for_flow):
                flow = compute_optical_flow(frame1_tensor, frame2_tensor)
                all_flow_fields.append(flow)
        
        for i, frame in enumerate(video_data):
            frame = frame.cpu().numpy()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_bgr = video_frame_decoration(frame_bgr, time_list[i], mode=mode, flow_field=all_flow_fields[i])
            
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            decorated.append(torch.from_numpy(frame_rgb))

        video_data = torch.stack(decorated, dim=0)
        video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        video_outputs = transform(video_data)

    elif video_decode_backend == 'opencv':
        cv2_vr = cv2.VideoCapture(video_path)
        duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if selected_indices is not None:
            frame_id_list = np.array(selected_indices, dtype=int)
        else:
            frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)

        fps = decord_vr.get_avg_fps() 
        time_list = [idx / fps for idx in frame_id_list]
        np.save('/home/scur0554/TempCompass/run_video_llava/time_list.npy', time_list)

        video_data = []
        for frame_idx in frame_id_list:
            cv2_vr.set(1, frame_idx)
            _, frame = cv2_vr.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_data.append(torch.from_numpy(frame).permute(2, 0, 1))
        cv2_vr.release()
        video_data = torch.stack(video_data, dim=1)
        video_outputs = transform(video_data)
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return video_outputs

class LanguageBindVideoProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = ("LanguageBindVideoTokenizer")

    def __init__(self, config, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.transform = get_video_transform(config)
        self.image_processor = load_and_transform_video
        self.tokenizer = tokenizer

    def __call__(self, images=None, text=None, context_length=77, selected_indices=None, return_tensors=None, **kwargs):
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            encoding = self.tokenizer(text, max_length=context_length, padding='max_length',
                                      truncation=True, return_tensors=return_tensors, **kwargs)

        if images is not None:
            images = make_list_of_images(images)
            image_features = [self.image_processor(image, self.transform,
                                                   video_decode_backend=self.config.vision_config.video_decode_backend,
                                                   num_frames=self.config.vision_config.num_frames, selected_indices=selected_indices) for image in images]
            image_features = torch.stack(image_features)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features
            return encoding
        elif text is not None:
            return encoding
        else:
            return {"pixel_values": image_features}

    def preprocess(self, images, return_tensors):
        return self.__call__(images=images, return_tensors=return_tensors)

    def batch_decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)

    def decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)
