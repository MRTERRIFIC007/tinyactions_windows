import cv2
import torch
import numpy as np

def load_all_frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret = True
    frames = []
    while ret:
        ret, frame = vidcap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    vidcap.release()
    assert len(frames) == frame_count
    frames = torch.from_numpy(np.stack(frames))
    return frames

frames = load_all_frames('/Users/mrterrific/Documents/Tinyactions/TinyActions/video.mp4')
print(frames.shape)

video_params = {
    "width" : 120,
    "height" : 120,
    "num_frames" : 16   # Reduced number of frames
}

def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255

pretrained_path = '/path/to/your/pretrained/weights.pth'

single_video_path = '/Users/mrterrific/Documents/Tinyactions/TinyActions/video.mp4'

if torch.cuda.is_available():
    print("Using CUDA...")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("Using MPS (Metal)....")
    device = torch.device("mps")
else:
    print("Using CPU....")
    device = torch.device("cpu")