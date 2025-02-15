import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import cv2
import numpy as np
import config as cfg
import Preprocessing

############ Helper Functions ##############
def resize(frames, size, interpolation='bilinear'):
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(frames.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(frames, size=size, scale_factor=scale, mode=interpolation, align_corners=False)

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)

def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255

class ToFloatTensorInZeroOne(object):
    def __call__(self, vid):
        return to_normalized_float_tensor(vid)

def normalize(vid, mean, std):
    shape = (-1,) + (1,) * (vid.dim() - 1)
    mean = torch.as_tensor(mean).reshape(shape)
    std = torch.as_tensor(std).reshape(shape)
    return (vid - mean) / std

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, vid):
        return normalize(vid, self.mean, self.std)


################# TinyVIRAT Dataset ###################
class TinyVIRAT_dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, IDs_path, labels, num_frames=cfg.video_params['num_frames'], input_size=cfg.video_params['height']):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.IDs_path = IDs_path
        self.num_frames = num_frames
        self.input_size = input_size
        self.resize = Resize((self.input_size, self.input_size))
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([ToFloatTensorInZeroOne(), self.resize, self.normalize])
        
        # If only one video is given, pre-load all its frames
        if len(self.list_IDs) == 1:
            video_path = self.IDs_path[self.list_IDs[0]]
            self.single_video_frames = self.load_all_frames(video_path)  # Expected shape: (T, H, W, C)
        else:
            self.single_video_frames = None

    def load_all_frames(self, video_path):
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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        vidcap.release()
        assert len(frames) == frame_count
        frames = torch.from_numpy(np.stack(frames))
        return frames

    def build_sample(self, video_path):
        frames = self.load_all_frames(video_path)
        count_frames = frames.shape[0]
        if count_frames > self.num_frames:
            frames = frames[:self.num_frames]
        elif count_frames < self.num_frames:  # Repeat last frame
            diff = self.num_frames - count_frames
            last_frame = frames[-1, :, :, :]
            tiled = np.tile(last_frame, (diff, 1, 1, 1))
            frames = np.append(frames, tiled, axis=0)
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)
        clips = self.transform(frames)
        return clips

    def __len__(self):
        'Denotes the total number of samples'
        if self.single_video_frames is not None:
            return self.single_video_frames.shape[0]
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # For a single video scenario, treat each frame as an individual sample.
        if self.single_video_frames is not None:
            # Get the frame at the particular index. self.single_video_frames is (T, H, W, C)
            frame = self.single_video_frames[index]
            # Wrap the frame in a new axis so it represents a "video" with 1 frame (T=1)
            frame = frame.unsqueeze(0)  # Now shape is (1, H, W, C)
            # Apply the same transformation chain that would be used for the full video.
            X = self.transform(frame)
            # Use the same label for every frame since it's from the single video.
            y = torch.Tensor(self.labels[self.list_IDs[0]])
            return X, y
        else:
            # Original implementation for multiple videos:
            ID = self.list_IDs[index]
            sample_path = self.IDs_path[ID]
            X = self.build_sample(sample_path)
            if len(self.labels) > 0:
                y = torch.Tensor(self.labels[ID])
            else:
                y = torch.Tensor([])
            return X, y

# Adjust for macOS Metal (MPS) device support
if torch.backends.mps.is_available():
    print("Using Metal (MPS) backend....")
    device = torch.device("mps")
else:
    print("Using CPU....")
    device = torch.device("cpu")

# Ensure tensors and models are moved to the appropriate device

# Add new dataset for loading a single video via a given path
class SingleVideoDataset(Dataset):
    """
    A Dataset for loading a single video file via a given path.
    Each sample corresponds to a single frame from the video,
    loaded on-demand when __getitem__ is called.
    """
    def __init__(self, video_path, input_size=cfg.video_params['height']):
        self.video_path = video_path
        self.input_size = input_size
        self.resize = Resize((self.input_size, self.input_size))
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([ToFloatTensorInZeroOne(), self.resize, self.normalize])
        # Open the video once to determine the total number of frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video: " + video_path)
        self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    def __len__(self):
        # Total number of samples equals the number of frames in the video.
        return self.num_frames

    def __getitem__(self, index):
        # Open the video file each time to read just the frame at the given index
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise Exception("Could not read frame at index " + str(index))
        # Convert BGR (OpenCV default) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame (H, W, C) to a tensor and add a frame dimension:
        frame = torch.from_numpy(frame)
        frame = frame.unsqueeze(0)  # Now shape is (1, H, W, C)
        # Apply the transformation pipeline
        X = self.transform(frame)
        # Create a dummy label (for example, a zero tensor of dimension [num_classes])
        y = torch.zeros(cfg.constants['num_classes'])
        return X, y
