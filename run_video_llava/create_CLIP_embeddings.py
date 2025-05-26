import cv2
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def load_video_as_pil_frames(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames is not None and count >= max_frames):
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        count += 1
    cap.release()
    return frames

def main():
    folder_path = "videos"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created directory: {folder_path}. Please add videos to process.")
        return

    mp4_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".mp4")]
    if not mp4_files:
        print(f"No .mp4 files found in '{folder_path}'.")

    output_dir = os.path.join("features")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    for f in mp4_files:
        video_path = os.path.join(folder_path, f)
        video_name = os.path.splitext(f)[0]
        feature_path = os.path.join(output_dir, f"{video_name}.pt")

        if os.path.exists(feature_path):
            print(f"Features for {f} already exist. Skipping.")
            continue

        try:
            print(f"Processing: {f}")
            pil_frames = load_video_as_pil_frames(video_path, max_frames=None) # Set max_frames if needed, e.g., 300

            if not pil_frames:
                print(f"Warning: No frames extracted for {f}. Skipping.")
                continue

            inputs = processor(images=pil_frames, return_tensors="pt", padding=True)
            # pixel_values = inputs['pixel_values'].to(device)
            
            batch_size = 32
            all_features = []

            for i in range(0, len(pil_frames), batch_size):
                batch_frames = pil_frames[i:i+batch_size]
                inputs = processor(images=batch_frames, return_tensors="pt", padding=True)
                pixel_values = inputs['pixel_values'].to(device)

                with torch.no_grad():
                    batch_features = model.get_image_features(pixel_values=pixel_values)
                    batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                    all_features.append(batch_features.cpu())
            image_features = torch.cat(all_features, dim=0)
            print(f"Extracted features shape: {image_features.shape}")

            torch.save(image_features, feature_path)

            # with torch.no_grad():
            #     image_features = model.get_image_features(pixel_values=pixel_values)
            #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            print(f"  Extracted features shape: {image_features.shape}")
            # torch.save(image_features.cpu(), feature_path)
            print(f"  Saved feature tensor for {f} to: {feature_path}")

        except Exception as e:
            print(f"-----------------------------------------")
            print(f"ERROR processing {f}: {e}")
            import traceback
            traceback.print_exc()
            print(f"-----------------------------------------")

if __name__ == "__main__":
    # Example: Create a dummy video for testing if 'videos' folder is empty
    videos_folder = "videos"
    if not os.path.exists(videos_folder):
        os.makedirs(videos_folder)
    if not any(f.lower().endswith(".mp4") for f in os.listdir(videos_folder)):
        print("Creating a dummy .mp4 video for testing...")
        try:
            # Attempt to create a short dummy mp4 file using opencv
            # This requires opencv-python and numpy
            import numpy as np
            dummy_video_path = os.path.join(videos_folder, "dummy_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(dummy_video_path, fourcc, 1, (64, 64)) # 1 fps, 64x64
            for _ in range(5): # 5 frames
                frame = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                out.write(frame)
            out.release()
            print(f"Created dummy video: {dummy_video_path}")
        except Exception as e:
            print(f"Could not create dummy video (ensure numpy and opencv-python are installed): {e}")
    main()