import os
import cv2
import numpy as np
import config as cfg
import json
import torch  # Ensure torch is imported for device handling

'''
Function: get_data_args
Args: 
    partn: Data partition [test,train,val]
    data_path: Path to  data folder having videos
    labels_path: Path to labels json file
Returns:
    list_IDs: A list of ids of samples in this folder
    labels: A dict of labels for each sample id
    IDs_path: A dict for mapping each sample to its path
'''

def get_data_args(partn, data_path, labels_path):
    with open(labels_path) as f:
        data = json.load(f)
    i = 0
    list_IDs = []
    labels = {}
    IDs_path = {}

    for sample in data:
        sample_id = 'id-' + str(i)
        list_IDs.append(sample_id)
        if partn != 'test':
            labels[sample_id] = get_labels(sample['label'])
        IDs_path[sample_id] = os.path.join(data_path, sample['path'])
        i += 1
    return list_IDs, labels, IDs_path

'''
Function to convert list of str labels to one hot encoding.
Arguments:
    labels: list of str labels
    class_map_path:
Output:
    onehot: list of one hot encoded labels
'''
def get_labels(labels, class_map_path=cfg.file_paths['class_map'], num_classes=cfg.constants['num_classes']):
    with open(class_map_path) as f:
        class_map = json.load(f)
    onehot = [0] * num_classes
    for lbl in labels:
        idx = class_map[lbl]
        onehot[idx] = 1
    return onehot

def get_prtn(prtn):
    if prtn == 'train':
        train_list_IDs, train_labels, train_IDs_path = get_data_args('train', cfg.file_paths['train_data'], cfg.file_paths['train_labels'])
        return train_list_IDs, train_labels, train_IDs_path
    elif prtn == 'val':
        val_list_IDs, val_labels, val_IDs_path = get_data_args('val', cfg.file_paths['val_data'], cfg.file_paths['val_labels'])
        return val_list_IDs, val_labels, val_IDs_path
    else:
        test_list_IDs, _, test_IDs_path = get_data_args('test', cfg.file_paths['test_data'], cfg.file_paths['test_labels'])
        return test_list_IDs, _, test_IDs_path

# Adjust for macOS Metal (MPS) device support
if torch.backends.mps.is_available():
    print("Using Metal (MPS) backend....")
    device = torch.device("mps")
else:
    print("Using CPU....")
    device = torch.device("cpu")

# You would move tensors and models to the correct device later in the workflow,
# such as during data loading, tensor creation, or model training like this:
# tensor = tensor.to(device)
# model = model.to(device)
