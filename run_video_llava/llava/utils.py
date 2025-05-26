import datetime
import logging
import logging.handlers
import os
import sys
import pickle
import numpy as np
import torch
from torch.nn import CosineSimilarity
import requests

from llava.constants import LOGDIR

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None
answer_prompt = {
    "multi-choice": "\nPlease directly give the best option:",
    "yes_no": "\nPlease answer yes or no:",
    "caption_matching": "\nPlease directly give the best option:",
    "captioning": ""
}

def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"

def load_frame_embeddings(video_embeddings_folder, video_path):
    video_id = os.path.basename(video_path).split('.')[0]
    path = os.path.join(video_embeddings_folder, video_id + '.pt') # Use os.path.join for robustness
    frame_embeddings = torch.load(path)
    return frame_embeddings
    
def load_query_embedding(all_query_embeddings_dict, current_inp, task_type): # Takes the loaded dict now
    """
    Retrieves a pre-computed query embedding from a dictionary.
    """
    task_specific_suffix = answer_prompt[task_type]
    original_question_text = current_inp

    if task_specific_suffix and current_inp.endswith(task_specific_suffix):
        original_question_text = current_inp[:-len(task_specific_suffix)]
    else:
        pass # Suffix is empty, current_inp is original

    if original_question_text in all_query_embeddings_dict:
        return all_query_embeddings_dict[original_question_text]
    else:
        print(f"Error: Original question text '{original_question_text}' not found as a key in query_embeddings_dict.")
        # For debugging:
        # print(f"Available keys (sample): {list(all_query_embeddings_dict.keys())[:5]}")
        return None

def inverse_transform_sampling(query_embedding, video_embeddings, n_samples, alpha=3):
    """
    Performs inverse transform sampling based on similarity scores.

    Args:
        query_embeddings (tensor):
        video_embeddings (tensor):
        n_samples (int): Number of frames to sample
        alpha (float): Sharpness parameter (typically 2.5 <= alpha <= 3.5)

    Returns:
        sampled_indices (np.ndarray): Indices of sampled frames
        cdf (np.ndarray): The calculated cumulative distribution function.
                          (Added this return for easier reuse if needed)
    """
    cosine_sim = CosineSimilarity(dim=1)
    
    similarity_scores = cosine_sim(video_embeddings, query_embedding)
    similarity_scores = similarity_scores.numpy()
    
    min_score = np.min(similarity_scores)
    max_score = np.max(similarity_scores)
    
    norm_scores = (similarity_scores - min_score) / (max_score - min_score)
    refined_scores = norm_scores ** alpha

    sum_refined_scores = np.sum(refined_scores)
    pdf = refined_scores / sum_refined_scores 
    cdf = np.cumsum(pdf)
    
    uniform_targets = np.linspace(0, 1, n_samples, endpoint=False) + (1.0 / (2 * n_samples))
    uniform_targets = np.clip(uniform_targets, 0, 1)

    sampled_indices = np.searchsorted(cdf, uniform_targets, side='left')
    sampled_indices = np.minimum(sampled_indices, len(similarity_scores) - 1)

    return sampled_indices
    

def select_gradient_boundaries(query_embedding, video_embeddings, n_samples=8, 
                           num_gradient_points=4, min_distance=15, boundary_offset=5, 
                           window_size=5):
    """
    Selects frames around the points with largest gradient in similarity curve.
    
    Args:
        query_embedding (tensor): CLIP embedding of the text query
        video_embeddings (tensor): CLIP embeddings of video frames
        n_samples (int): Total number of frames to return
        num_gradient_points (int): Number of gradient points to identify (default: 4)
        min_distance (int): Minimum distance between gradient points (default: 15)
        boundary_offset (int): Number of frames to look before/after gradient points (default: 5)
        window_size (int): Size of smoothing window (default: 5)
        
    Returns:
        sampled_indices (np.ndarray): Indices of sampled frames
    """
    from scipy.ndimage import gaussian_filter1d
    
    # Compute similarity scores
    cosine_sim = CosineSimilarity(dim=1)
    similarity_scores = cosine_sim(video_embeddings, query_embedding).numpy()
    
    # Apply smoothing to reduce noise
    smoothed = gaussian_filter1d(similarity_scores, sigma=window_size / 3.0)
    
    # Calculate gradient
    gradient = np.gradient(smoothed)
    abs_gradient = np.abs(gradient)
    
    # Sort indices by gradient magnitude (descending)
    sorted_indices = np.argsort(-abs_gradient)
    
    # Select largest gradient points with minimum distance constraint
    gradient_points = []
    for idx in sorted_indices:
        # Check if this point is far enough from already selected points
        if all(abs(idx - point) >= min_distance for point in gradient_points):
            gradient_points.append(idx)
            
        # Stop once we have enough points
        if len(gradient_points) >= num_gradient_points:
            break
    
    # For each gradient point, select frames before and after
    selected_frames = []
    for point in gradient_points:
        # Add frame before the gradient point (if within bounds)
        before_idx = max(0, point - boundary_offset)
        selected_frames.append(before_idx)
        
        # Add frame after the gradient point (if within bounds)
        after_idx = min(len(similarity_scores) - 1, point + boundary_offset)
        selected_frames.append(after_idx)
    
    # If we have more frames than requested, trim the list
    if len(selected_frames) > n_samples:
        selected_frames = selected_frames[:n_samples]
    
    # If we have fewer frames than requested, add frames using similarity scores
    if len(selected_frames) < n_samples:
        # Create a mask to exclude already selected frames
        mask = np.ones(len(similarity_scores), dtype=bool)
        for idx in selected_frames:
            mask[idx] = False
        
        # Get the remaining frames and their similarity scores
        remaining_indices = np.arange(len(similarity_scores))[mask]
        remaining_scores = similarity_scores[mask]
        
        if len(remaining_indices) > 0:
            # Sort by similarity score (descending)
            sorted_remaining = remaining_indices[np.argsort(-remaining_scores)]
            
            # Add the frames with highest similarity scores
            additional_needed = n_samples - len(selected_frames)
            additional_frames = sorted_remaining[:additional_needed]
            
            selected_frames.extend(additional_frames)
    
    # Remove any duplicates while preserving order
    unique_frames = []
    for idx in selected_frames:
        if idx not in unique_frames:
            unique_frames.append(idx)
    
    return np.array(unique_frames)



    
    
    
    
    
    
    
