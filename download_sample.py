#!/usr/bin/env python3
"""
Script to download a sample video if one doesn't exist.
This is useful for environments where the sample video is not available.
"""

import os
import sys
import urllib.request
import config as cfg

def download_sample_video():
    """Download a sample video if one doesn't exist."""
    # Destination path in the training directory
    dest_dir = cfg.file_paths['train_data']
    dest_path = os.path.join(dest_dir, 'sample_video.mp4')
    
    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Check if the video already exists
    if os.path.exists(dest_path):
        print(f"Sample video already exists at {dest_path}")
        return True
    
    # URLs of sample videos to try
    sample_urls = [
        "https://filesamples.com/samples/video/mp4/sample_640x360.mp4",
        "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4",
        "https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
    ]
    
    # Try to download from each URL
    for url in sample_urls:
        try:
            print(f"Downloading sample video from {url}...")
            urllib.request.urlretrieve(url, dest_path)
            print(f"Sample video downloaded to {dest_path}")
            return True
        except Exception as e:
            print(f"Error downloading from {url}: {e}")
    
    print("Failed to download sample video from any URL.")
    return False

if __name__ == "__main__":
    success = download_sample_video()
    if success:
        print("Download completed successfully. You can now run train.py")
    else:
        print("Download failed. Please check the error messages above.") 