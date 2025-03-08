import os

# Get base directory from environment variable or use a default relative path
# If the script is at Tinyactions/tinyactions_2/tinyactions_windows/TinyVIRAT_V2
# we need to set the base directory correctly
script_dir = os.path.dirname(os.path.abspath(__file__))
default_base_dir = os.path.join(script_dir, 'TinyVIRAT_V2')

# Additional possible dataset locations
additional_dataset_paths = [
    '/workspace/TinyVIRAT_V2',  # Specified path
    '/workspace/tinyactions_2/tinyactions_windows/TinyVIRAT_V2',
    '/workspace/Tinyactions/TinyActions wins/TinyVIRAT_V2',
]

# If we're already in a TinyVIRAT_V2 directory, use the parent directory
if os.path.basename(script_dir) == 'TinyVIRAT_V2':
    default_base_dir = script_dir

# Try to find the dataset in the additional paths
BASE_DIR = os.environ.get('TINYACTIONS_DATA_DIR', default_base_dir)

# Check if the directory exists, if not, try the additional paths
if not os.path.exists(BASE_DIR):
    print(f"Warning: Directory {BASE_DIR} not found. Checking alternative locations...")
    for path in additional_dataset_paths:
        if os.path.exists(path):
            BASE_DIR = path
            print(f"Found dataset at {BASE_DIR}")
            break
    else:  # This else belongs to the for loop, executes if no break occurred
        print(f"Warning: Dataset not found in any of the expected locations. Using current directory as fallback.")
        BASE_DIR = script_dir

video_params = {
    "width": 120,
    "height": 120,
    "num_frames": 60
}

constants = {
    "num_classes": 26
}

# Create the videos directory paths even if they don't exist
train_videos_dir = os.path.join(BASE_DIR, 'videos', 'train')
val_videos_dir = os.path.join(BASE_DIR, 'videos', 'val')
test_videos_dir = os.path.join(BASE_DIR, 'videos', 'test')

# Ensure the directories exist (create them if they don't)
os.makedirs(train_videos_dir, exist_ok=True)
os.makedirs(val_videos_dir, exist_ok=True)
os.makedirs(test_videos_dir, exist_ok=True)

file_paths = {
    'train_data': train_videos_dir,
    'train_labels': os.path.join(BASE_DIR, 'tiny_train_v2.json'),
    'val_data': val_videos_dir,
    'val_labels': os.path.join(BASE_DIR, 'tiny_val_v2.json'),
    'test_data': test_videos_dir,
    'test_labels': os.path.join(BASE_DIR, 'tiny_test_v2_public.json'),
    'class_map': os.path.join(BASE_DIR, 'class_map.json')
}

# Print configuration for debugging
if __name__ == "__main__":
    print(f"Script directory: {script_dir}")
    print(f"Base directory: {BASE_DIR}")
    for key, path in file_paths.items():
        print(f"{key}: {path}")
        print(f"  Exists: {os.path.exists(path)}")

