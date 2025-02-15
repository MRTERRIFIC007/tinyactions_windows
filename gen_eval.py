import torch
import numpy as np
from Model.VideoSWIN import VideoSWIN3D
from configuration import build_config
from dataloader2 import TinyVirat
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

exp = 'e22_val'

# Make exp dir
if not os.path.exists('evals/' + exp + '/'):
    os.makedirs('evals/' + exp + '/')
PATH = 'evals/' + exp + '/'

def compute_labels(pred, inf_th):
    # Pass pred through sigmoid
    pred = torch.sigmoid(pred)
    pred = pred.cpu().data.numpy()

    # Use inference threshold to get one-hot encoded labels
    print("Predictions: ", pred)
    res = pred > inf_th
    pred = list(map(int, res[0]))  # Convert to list of integers
    
    return pred

# Determine if MPS (Metal) is available
if torch.backends.mps.is_available():
    print("Using Metal (MPS) backend....")
    device = torch.device("mps")
else:
    print("Using CPU....")
    device = torch.device("cpu")

# Training Parameters
shuffle = True
print("Creating params....")
params = {
    'batch_size': 1,
    'shuffle': shuffle,
    'num_workers': 2
}

inf_threshold = 0.5

# Data Generators
cfg = build_config('TinyVirat')
dataset = TinyVirat(cfg=cfg, data_split='test')
test_generator = DataLoader(dataset, **params)

# Define model
print("Initiating Model...")

ckpt_path = '/home/mo926312/Documents/TinyActions/Slurm_Scripts/exps/exp_22/22_best_ckpt.pt'
model = VideoSWIN3D()
model.load_state_dict(torch.load(ckpt_path, map_location=device))  # Load model on device
model = model.to(device)

count = 0
print("Begin Evaluation....")
model.eval()
rmap = {}

with open('answer.txt', 'w') as wid:
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_generator)):
            inputs = inputs.to(device)

            # Squeeze clips dimension for dataloader 2
            inputs = torch.squeeze(inputs, dim=1)
            video_id = targets[0]['path'][0].split('.')[0]

            predictions = model(inputs.float())

            # Get predicted labels for this video sample
            labels = compute_labels(predictions, inf_threshold)

            # Format result string for this video
            str_labels = " ".join(map(str, labels))
            result_string = f"{video_id} {str_labels}"

            # Add result to rmap
            rmap[video_id] = result_string
            count += 1

    # Add remaining video ID labels in the answer.txt
    for id in range(6097):
        vid_id = str(id).zfill(5)
        if vid_id not in rmap:
            result_string = f"{vid_id} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
        else:
            result_string = rmap[vid_id]
        print("Result String: ", result_string)
        wid.write(result_string + '\n')

print(f"Total Samples: {count}")
