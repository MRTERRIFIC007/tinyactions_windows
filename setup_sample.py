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
    cwd = os.getcwd()
    
    possible_paths = [
        os.path.join(script_dir, 'video.mp4'),  # In the same directory as this script
        os.path.join(script_dir, '..', 'video.mp4'),  # One level up
        os.path.join(script_dir, '..', '..', 'video.mp4'),  # Two levels up
        os.path.join(cwd, 'video.mp4'),  # In the current working directory
        os.path.join(cwd, '..', 'video.mp4'),  # One level up from cwd
        '/workspace/video.mp4',  # Root of workspace (common in Docker/Jupyter)
        '/workspace/Tinyactions_2/video.mp4',  # Common Docker/Jupyter path
        '/workspace/Tinyactions_2/tinyactions_windows/video.mp4',  # Specific to your environment
    ]
    
    # If we can't find the video, try to create a dummy video
    sample_video_path = None
    for path in possible_paths:
        if os.path.exists(path):
            sample_video_path = path
            break
    
    # If we still can't find the video, try to create a dummy video
    if not sample_video_path:
        print("Sample video not found in any of the expected locations.")
        print("Attempting to create a dummy video...")
        try:
            import numpy as np
            import cv2
            
            # Create a dummy video file
            dummy_path = os.path.join(script_dir, 'dummy_video.mp4')
            
            # Create a blank video with a moving rectangle
            fps = 30
            width, height = 120, 120
            duration = 3  # seconds
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(dummy_path, fourcc, fps, (width, height))
            
            for i in range(fps * duration):
                # Create a blank frame
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Draw a moving rectangle
                x = int(width * (i / (fps * duration)))
                cv2.rectangle(frame, (x, 40), (x + 40, 80), (0, 255, 0), -1)
                
                # Write the frame
                out.write(frame)
            
            out.release()
            
            sample_video_path = dummy_path
            print(f"Created dummy video at {dummy_path}")
        except Exception as e:
            print(f"Error creating dummy video: {e}")
    
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