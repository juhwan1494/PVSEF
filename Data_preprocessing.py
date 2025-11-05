import os
import av
import glob
import numpy as np
from tqdm import tqdm
import cv2
from transformers import VivitImageProcessor

np.random.seed(0)  

def read_video_pyav(container, indices, target_size=(224, 224), video_path="unknown"):
    frames = []
    container.seek(0)  
    start_index = indices[0] 
    end_index = indices[-1]   
    try:
        for i, frame in enumerate(container.decode(video=0)):  
            if i > end_index:  
                break
            if i >= start_index and i in indices:  
                try:
                    frame_array = frame.to_ndarray(format="rgb24") 
                    resized_frame = cv2.resize(frame_array, target_size)  
                    frames.append(resized_frame) 
                except Exception as e:
                    print(f"Error decoding frame {i} in {video_path}: {e}")
    except Exception as e:
        print(f"Error decoding video {video_path}: {e}")
    
    if len(frames) == 0:
        print(f"Error: No frames extracted for indices {indices} in {video_path}. Using default frame.")
        default_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frames = [default_frame] * len(indices)
    
    elif len(frames) < len(indices):
        print(f"Warning: Missing frames in {video_path}. Extracted {len(frames)} frames, expected {len(indices)}.")
        while len(frames) < len(indices):
            frames.append(frames[-1])  
    
    return np.stack(frames)  


def sample_frame_indices(clip_len, start_len, seg_len):
    frame_start = start_len
    frame_end = max(start_len, seg_len - 1)  
    if frame_end - frame_start + 1 < clip_len:
        indices = np.arange(frame_start, frame_end + 1)
        indices = np.repeat(indices, clip_len // len(indices) + 1)[:clip_len]
    else:
        indices = np.linspace(frame_start, frame_end, num=clip_len)
        indices = np.clip(indices, frame_start, frame_end).astype(np.int64)
    return indices

video_datas = glob.glob('/your_directory/*.avi')  
image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")  
savedir = '/your_directory/preprocessed_dataset/'  

valid_videos = []
for data in video_datas:
    try:
        container = av.open(data)  
        total_frames = container.streams.video[0].frames 
        if total_frames > 0:
            valid_videos.append(data)
        else:
            print(f"Warning: {data} has 0 frames. Skipping.")
        container.close()
    except Exception as e:
        print(f"Error: Failed to open {data}: {e}. Skipping.")

for data in tqdm(valid_videos):
    container = av.open(data)
    total_frames = container.streams.video[0].frames  
    base_name = os.path.basename(data).replace('.avi', '')  
    patient_folder = os.path.join(savedir, base_name)  
    os.makedirs(patient_folder, exist_ok=True)
    
    clip_len = 32  
    num_segments = (total_frames + clip_len - 1) // clip_len  

    for i in range(num_segments):
        start_len = i * clip_len  
        end_len = min((i + 1) * clip_len, total_frames)  
        
        indices = sample_frame_indices(clip_len=clip_len, start_len=start_len, seg_len=end_len)  
        print(f"Processing {data}, segment {i+1}, indices: {indices}")
        video = read_video_pyav(container=container, indices=indices, target_size=(224, 224), video_path=data)
        
        if video.shape[0] < clip_len:
            print(f"Padding video for {data}, segment {i+1}. Current frames: {video.shape[0]}")
            while video.shape[0] < clip_len:
                video = np.append(video, [video[-1]], axis=0)
        
        
        video = image_processor(list(video), return_tensors="pt")
        sampling_av = video['pixel_values'][0].numpy()  # numpy 변환
        
        
        save_path = os.path.join(patient_folder, f'{base_name}_{i+1}.npy')
        np.save(save_path, sampling_av)

print("Processing complete.")
