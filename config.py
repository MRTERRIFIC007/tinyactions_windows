import os

# Get base directory from environment variable or use a default relative path
BASE_DIR = os.environ.get('TINYACTIONS_DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TinyVIRAT_V2'))

video_params = {
    "width": 120,
    "height": 120,
    "num_frames": 60
}

constants = {
    "num_classes": 26
}

file_paths = {
    'train_data': os.path.join(BASE_DIR, 'videos', 'train'),
    'train_labels': os.path.join(BASE_DIR, 'tiny_train_v2.json'),
    'val_data': os.path.join(BASE_DIR, 'videos', 'val'),
    'val_labels': os.path.join(BASE_DIR, 'tiny_val_v2.json'),
    'test_data': os.path.join(BASE_DIR, 'videos', 'test'),
    'test_labels': os.path.join(BASE_DIR, 'tiny_test_v2_public.json'),
    'class_map': os.path.join(BASE_DIR, 'class_map.json')
}

# Print configuration for debugging
if __name__ == "__main__":
    print(f"Base directory: {BASE_DIR}")
    for key, path in file_paths.items():
        print(f"{key}: {path}")
        print(f"  Exists: {os.path.exists(path)}")

