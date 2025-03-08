import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import cv2
import numpy as np
import config as cfg
import Preprocessing
import os
import random

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

# Data augmentation transforms
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, vid):
        if random.random() < self.p:
            return torch.flip(vid, [3])  # Flip width dimension
        return vid

class RandomColorJitter(object):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        
    def __call__(self, vid):
        # Apply color jitter to each frame
        # vid shape: C, T, H, W
        C, T, H, W = vid.shape
        
        # Randomly adjust brightness, contrast, saturation, hue
        brightness_factor = random.uniform(1-self.brightness, 1+self.brightness)
        contrast_factor = random.uniform(1-self.contrast, 1+self.contrast)
        saturation_factor = random.uniform(1-self.saturation, 1+self.saturation)
        hue_factor = random.uniform(-self.hue, self.hue)
        
        # Apply to each frame
        result = vid.clone()
        for t in range(T):
            frame = vid[:, t, :, :]  # C, H, W
            
            # Brightness
            frame = frame * brightness_factor
            
            # Contrast
            mean = torch.mean(frame, dim=[1, 2], keepdim=True)
            frame = (frame - mean) * contrast_factor + mean
            
            # Clamp values to [0, 1]
            frame = torch.clamp(frame, 0, 1)
            
            result[:, t, :, :] = frame
            
        return result

class RandomCrop(object):
    def __init__(self, size):
        self.size = size
        
    def __call__(self, vid):
        # vid shape: C, T, H, W
        C, T, H, W = vid.shape
        
        # Calculate crop dimensions
        new_h, new_w = self.size, self.size
        
        # Don't crop if the video is already smaller than the crop size
        if H <= new_h or W <= new_w:
            return vid
            
        # Random crop position
        top = random.randint(0, H - new_h)
        left = random.randint(0, W - new_w)
        
        # Crop the video
        return vid[:, :, top:top+new_h, left:left+new_w]

class RandomRotation(object):
    def __init__(self, degrees=10):
        self.degrees = degrees
        
    def __call__(self, vid):
        # vid shape: C, T, H, W
        C, T, H, W = vid.shape
        
        # Random rotation angle
        angle = random.uniform(-self.degrees, self.degrees)
        
        # Apply rotation to each frame
        result = torch.zeros_like(vid)
        for t in range(T):
            frame = vid[:, t, :, :]  # C, H, W
            
            # Convert to numpy for OpenCV
            frame_np = frame.permute(1, 2, 0).numpy()  # H, W, C
            
            # Get rotation matrix
            center = (W // 2, H // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation
            rotated = cv2.warpAffine(frame_np, M, (W, H))
            
            # Convert back to tensor
            result[:, t, :, :] = torch.from_numpy(rotated).permute(2, 0, 1)
            
        return result

################# TinyVIRAT Dataset ###################
class TinyVIRAT_dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, IDs_path, labels, num_frames=cfg.video_params['num_frames'], input_size=cfg.video_params['height'], frame_by_frame=False, use_augmentation=True, is_training=True):
        "Initialization"
        self.labels = labels
        self.list_IDs = list_IDs
        self.IDs_path = IDs_path
        self.num_frames = num_frames
        self.input_size = input_size
        self.frame_by_frame = frame_by_frame
        self.is_training = is_training
        self.use_augmentation = use_augmentation and is_training
        
        # Basic transforms
        self.resize = Resize((self.input_size, self.input_size))
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Create transform pipeline
        if self.use_augmentation:
            print("Using data augmentation for training")
            self.transform = transforms.Compose([
                ToFloatTensorInZeroOne(),
                self.resize,
                RandomHorizontalFlip(p=0.5),
                RandomColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                RandomRotation(degrees=10),
                self.normalize
            ])
        else:
            self.transform = transforms.Compose([
                ToFloatTensorInZeroOne(),
                self.resize,
                self.normalize
            ])
        
        # For single video scenario, load all frames
        if len(self.list_IDs) == 1:
            video_path = self.IDs_path[self.list_IDs[0]]
            self.single_video_frames = self.load_all_frames(video_path)
            self.frame_by_frame = False
            self.frame_index_map = None
        else:
            self.single_video_frames = None
            if self.frame_by_frame:
                # Build a mapping from a global sample index to (video_id, frame_index)
                self.frame_index_map = []
                for vid in self.list_IDs:
                    path = self.IDs_path[vid]
                    cap = cv2.VideoCapture(path)
                    if not cap.isOpened():
                        print(f"Warning: Could not open video {vid}")
                        continue
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    for i in range(frame_count):
                        self.frame_index_map.append((vid, i))
            else:
                self.frame_index_map = None

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
        if self.frame_index_map is not None:
            return len(self.frame_index_map)
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        if self.single_video_frames is not None:
            # For single video case, each frame is a sample.
            frame = self.single_video_frames[index]
            frame = frame.unsqueeze(0)
            X = self.transform(frame)
            X = X.to(device)  # move tensor to GPU if available
            y = torch.Tensor(self.labels[self.list_IDs[0]])
            return X, y
        elif self.frame_index_map is not None:
            # Frame-by-frame mode for multiple videos.
            vid, frame_num = self.frame_index_map[index]
            video_path = self.IDs_path[vid]
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise Exception(f"Could not read frame {frame_num} from video {vid}")
            # Convert frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert frame to tensor (H, W, C)
            frame = torch.from_numpy(frame)
            # Unsqueeze to add temporal dimension: (1, H, W, C)
            frame = frame.unsqueeze(0)
            X = self.transform(frame)
            X = X.to(device)  # move tensor to GPU if available
            y = torch.Tensor(self.labels[vid])
            return X, y
        else:
            # Default: treat each video as one sample (build a clip)
            ID = self.list_IDs[index]
            if index % 100 == 0:
                print(f"Loading video {index}: {ID}")
            sample_path = self.IDs_path[ID]
            X = self.build_sample(sample_path)
            X = X.to(device)  # move tensor to GPU if available
            y = torch.Tensor(self.labels[ID])
            return X, y

# Adjust for CUDA, MPS, or CPU device support
try:
    if torch.cuda.is_available():
        print("Using CUDA....")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using Metal (MPS) backend....")
        device = torch.device("mps")
    else:
        print("Using CPU....")
        device = torch.device("cpu")
except Exception as e:
    print("Error in device selection, defaulting to CPU:", e)
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
        X = X.to(device)  # move tensor to GPU if available
        # Create a dummy label (for example, a zero tensor of dimension [num_classes])
        y = torch.zeros(cfg.constants['num_classes'])
        return X, y

# Add the helper function to recursively obtain video file paths and dummy labels
def get_video_data(root_path, num_classes=26):
    """
    Recursively collect video file paths from the given root folder.
    
    Args:
        root_path (str): Path to the folder containing training videos (and subfolders).
        num_classes (int): Number of classes. Dummy labels will be vectors of zeros.
        
    Returns:
        list_IDs (list): Unique IDs for each video (based on relative paths).
        labels (dict): Mapping from video ID to dummy labels (zero vector of length num_classes).
        IDs_path (dict): Mapping from video ID to its full file path.
    """
    list_IDs = []
    IDs_path = {}
    labels = {}
    
    # Check if the directory exists
    if not os.path.exists(root_path):
        print(f"Warning: Directory {root_path} does not exist.")
        return list_IDs, labels, IDs_path
    
    for subdir, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.mp4'):
                full_path = os.path.join(subdir, file)
                # Use the relative path as a unique video ID
                video_id = os.path.relpath(full_path, root_path)
                list_IDs.append(video_id)
                IDs_path[video_id] = full_path
                labels[video_id] = [0] * num_classes
    
    return list_IDs, labels, IDs_path

if __name__ == '__main__':
    # Optionally test the get_video_data function
    video_root = cfg.file_paths['train_data']
    list_IDs, labels, IDs_path = get_video_data(video_root)
    print(f"Found {len(list_IDs)} video(s) in {video_root}")
