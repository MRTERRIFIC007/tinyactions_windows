#!/usr/bin/env python3
"""
Setup script to copy the sample video to the training directory if needed.
This is useful for environments where the full dataset is not available.
"""

import os
import shutil
import config as cfg

def setup_sample_video():
    """Copy the sample video to the training directory if it doesn't exist."""
    # Try multiple possible locations for the sample video
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(script_dir, 'video.mp4'),  # In the same directory as this script
        os.path.join(script_dir, '..', 'video.mp4'),  # One level up
        os.path.join(script_dir, '..', '..', 'video.mp4'),  # Two levels up
    ]
    
    sample_video_path = None
    for path in possible_paths:
        if os.path.exists(path):
            sample_video_path = path
            break
    
    # Check if the source video exists
    if not sample_video_path:
        print(f"Error: Sample video not found in any of these locations:")
        for path in possible_paths:
            print(f"  - {path}")
        return False
    
    # Destination path in the training directory
    dest_dir = cfg.file_paths['train_data']
    dest_path = os.path.join(dest_dir, 'sample_video.mp4')
    
    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Copy the video if it doesn't already exist in the destination
    if not os.path.exists(dest_path):
        try:
            shutil.copy2(sample_video_path, dest_path)
            print(f"Sample video copied from {sample_video_path} to {dest_path}")
            return True
        except Exception as e:
            print(f"Error copying sample video: {e}")
            return False
    else:
        print(f"Sample video already exists at {dest_path}")
        return True

if __name__ == "__main__":
    success = setup_sample_video()
    if success:
        print("Setup completed successfully. You can now run train.py")
    else:
        print("Setup failed. Please check the error messages above.") 