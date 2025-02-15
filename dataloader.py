import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import cv2
import random
import numpy as np
from configuration import build_config
from tqdm import tqdm
import time

# Constants
VIDEO_LENGTH = 100  # Number of frames in each video
TUBELET_TIME = VIDEO_LENGTH  # Duration of each tubelet
NUM_CLIPS = VIDEO_LENGTH // TUBELET_TIME

# Check if Metal (MPS) is available, fallback to CPU if not
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

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

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class TinyVirat(Dataset):
    def __init__(self, cfg, data_split, data_percentage, num_frames, skip_frames, input_size, shuffle=False):
        self.data_split = data_split
        self.num_classes = cfg.num_classes
        self.class_labels = [k for k, v in sorted(json.load(open(cfg.class_map, 'r')).items(), key=lambda item: item[1])]
        assert data_split in ['train', 'val', 'test']
        if data_split == 'train':
            annotations = json.load(open(cfg.train_annotations, 'r'))
        elif data_split == 'val':
            annotations = json.load(open(cfg.val_annotations, 'r'))
        else:
            annotations = json.load(open(cfg.test_annotations, 'r'))
        self.data_folder = os.path.join(cfg.data_folder, data_split)
        self.annotations  = {}
        for annotation in annotations:
            if annotation['dim'][0] < num_frames:
                continue
            if annotation['id'] not in self.annotations:
                self.annotations[annotation['id']] = {}
            self.annotations[annotation['id']]['path'] = annotation['path']
            if data_split == 'test':
                self.annotations[annotation['id']]['label'] = []
            else:
                self.annotations[annotation['id']]['label'] = annotation['label']
            self.annotations[annotation['id']]['length'] = annotation['dim'][0]
            self.annotations[annotation['id']]['width'] = annotation['dim'][1]
            self.annotations[annotation['id']]['height'] = annotation['dim'][2]
        self.video_ids = list(self.annotations.keys())
        if shuffle:
            random.shuffle(self.video_ids)
        len_data = int(len(self.video_ids) * data_percentage)
        self.video_ids = self.video_ids[0:len_data]
        
        # Reduce the number of frames
        self.num_frames = min(num_frames, 16)  # Limit to 16 frames
        
        # Reduce input size
        self.input_size = min(input_size, 112)  # Limit to 112x112 resolution
        
        self.resize = Resize((self.input_size, self.input_size))
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([ToFloatTensorInZeroOne(), self.resize, self.normalize])

    def __len__(self):
        return len(self.video_ids)

    def load_frames_random(self, video_path):
        vidcap = cv2.VideoCapture(video_path)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        skip_frames = self.skip_frames
        while frame_count < self.num_frames * skip_frames:
            if skip_frames <= 1:
                skip_frames = 1
                break
            skip_frames = skip_frames // 2
        assert frame_count >= self.num_frames * skip_frames
        random_start = random.randint(0, frame_count - self.num_frames * skip_frames)
        frame_indices = [indx for indx in range(random_start, random_start + self.num_frames * skip_frames, skip_frames)]
        ret = True
        counter = 0
        frames = []
        while ret:
            ret, frame = vidcap.read()
            if counter > max(frame_indices):
                ret = False
            if counter in frame_indices:
                frame = cv2.resize(frame, (self.input_size, self.input_size))
                frames.append(frame)
                counter += 1
            else:
                counter += 1
                continue
        vidcap.release()
        assert len(frames) == self.num_frames
        frames = torch.from_numpy(np.stack(frames)).to(device)  # Move frames to Metal device
        return frames

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
        frames = torch.from_numpy(np.stack(frames)).to(device)  # Move frames to Metal device
        return frames

    def build_random_clip(self, video_path):
        frames = self.load_frames_random(video_path)
        frames = self.transform(frames)
        return frames

    def build_consecutive_clips(self, video_path):
        frames = self.load_all_frames(video_path)
        if len(frames) % self.num_frames != 0:
            frames = frames[:len(frames) - (len(frames) % self.num_frames)]
        
        # Reduce number of clips
        max_clips = 4  # Limit to 4 clips
        clips = torch.stack([self.transform(x) for x in chunks(frames[:self.num_frames * max_clips], self.num_frames)]).to(device)
        
        return clips
    
    def __getitem__(self, index):
        video_id = self.video_ids[index]
        video_path = os.path.join(self.data_folder, self.annotations[video_id]['path'])
        video_len = self.annotations[video_id]['length']
        
        if self.data_split == 'test':
            video_labels = []
        else:
            video_labels = self.annotations[video_id]['label']
        clips = self.build_consecutive_clips(video_path)
           
        label = np.zeros(self.num_classes)
        for _class in video_labels:
            label[self.class_labels.index(_class)] = 1
        
        # Ensure the sample has at most 4 clips
        if clips.shape[0] > 4:
            clips = clips[:4,:,:,:,:]
        
        if self.data_split == 'test':
            return clips, [self.annotations[video_id]]
                
        return clips, label  # clips: nc x ch x t x H x W

'''
# Example usage
if __name__ == '__main__':
    shuffle = True
    batch_size = 1

    dataset = 'TinyVirat'
    cfg = build_config(dataset)

    data_generator = TinyVirat(cfg, 'test', 1.0, num_frames=32, skip_frames=2, input_size=224, shuffle=shuffle)

    generator = DataLoader(data_generator, batch_size=batch_size, num_workers=1, shuffle=False)
    
    for (clips, label) in generator:
        print(clips.shape, label)
'''

class TinyVIRAT_dataset(Dataset):
    def __init__(self, list_IDs, labels, IDs_path, num_frames=8, input_size=64):
        self.list_IDs = list_IDs
        self.labels = labels
        self.IDs_path = IDs_path
        self.num_frames = num_frames
        self.input_size = input_size

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        video_path = self.IDs_path[ID]
        
        frames = self.load_video(video_path)
        
        if self.labels:
            label = torch.tensor(self.labels[ID], dtype=torch.float32)
        else:
            label = torch.tensor([0] * 26, dtype=torch.float32)  # Assuming 26 classes
        
        return frames, label

    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        while len(frames) < self.num_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % 4 == 0:  # Skip every 4 frames
                frame = cv2.resize(frame, (self.input_size, self.input_size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                frame = torch.from_numpy(frame).unsqueeze(0).float() / 255.0
                frames.append(frame)
            
            frame_count += 1

        cap.release()

        # If we don't have enough frames, loop the video
        while len(frames) < self.num_frames:
            frames.extend(frames[:self.num_frames - len(frames)])

        return torch.stack(frames[:self.num_frames])
