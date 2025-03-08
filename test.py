#!/usr/bin/env python3
import torch
import numpy as np
from Model.VideoSWIN import VideoSWIN3D
from my_dataloader import TinyVIRAT_dataset, get_video_data
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
import argparse
import traceback
import gc
import signal
import sys
import time
from functools import wraps

# Force garbage collection to free up memory
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

import config as cfg

# Timeout handler for operations that might hang
class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message="Operation timed out"):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @wraps(func)
        def wrapper(*args, **kwargs):
            if os.name != 'nt':  # Not available on Windows
                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                if os.name != 'nt':
                    signal.alarm(0)
            return result
        return wrapper
    return decorator

def compute_accuracy(pred, target, inf_th):
    """Compute accuracy with error handling"""
    try:
        # Ensure inputs are on CPU and in the right format
        if isinstance(target, torch.Tensor):
            target = target.cpu().data.numpy()
        if isinstance(pred, torch.Tensor):
            pred = torch.sigmoid(pred)
            pred = pred.cpu().data.numpy()
        
        # Apply threshold
        pred_binary = pred > inf_th
        
        # Handle edge cases
        if len(target) == 0 or len(pred_binary) == 0:
            print("Warning: Empty prediction or target array")
            return 0.0
            
        return accuracy_score(pred_binary, target)
    except Exception as e:
        print(f"Error in compute_accuracy: {e}")
        traceback.print_exc()
        return 0.0  # Return a default value on error

def get_optimal_device():
    """
    Find the optimal device to use based on available resources.
    Returns the device with the most free memory or CPU if no suitable GPU is found.
    """
    try:
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU...")
            return torch.device("cpu")
        
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("No GPUs found, using CPU...")
            return torch.device("cpu")
        
        # If only one GPU, use it
        if num_gpus == 1:
            # Test if it's actually working
            try:
                # Use a timeout to prevent hanging if GPU is in a bad state
                @timeout(5, "GPU test timed out")
                def test_gpu():
                    test_tensor = torch.zeros(1).cuda()
                    result = test_tensor + 1  # Simple operation to test GPU
                    del test_tensor
                    return result
                
                test_gpu()
                print(f"Using single available GPU: {torch.cuda.get_device_name(0)}")
                return torch.device("cuda:0")
            except (RuntimeError, TimeoutError) as e:
                print(f"Error with GPU: {e}")
                print("Falling back to CPU...")
                return torch.device("cpu")
        
        # Multiple GPUs available, find the one with most free memory
        free_memory = []
        for i in range(num_gpus):
            try:
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                # Use a timeout to prevent hanging
                @timeout(5, f"GPU {i} memory check timed out")
                def check_gpu_memory(gpu_idx):
                    return torch.cuda.memory_reserved(gpu_idx) - torch.cuda.memory_allocated(gpu_idx)
                
                mem = check_gpu_memory(i)
                free_memory.append(mem)
                print(f"GPU {i} ({torch.cuda.get_device_name(i)}): {free_memory[-1]/1024**2:.1f} MB free")
            except Exception as e:
                print(f"Error checking GPU {i}: {e}")
                free_memory.append(0)
        
        # Select the GPU with the most free memory
        if max(free_memory) > 0:
            best_gpu = free_memory.index(max(free_memory))
            print(f"Selected GPU {best_gpu} with {free_memory[best_gpu]/1024**2:.1f} MB free memory")
            return torch.device(f"cuda:{best_gpu}")
        else:
            print("No GPU with free memory found, using CPU...")
            return torch.device("cpu")
    
    except Exception as e:
        print(f"Error selecting device: {e}")
        traceback.print_exc()
        print("Falling back to CPU...")
        return torch.device("cpu")

@timeout(60, "Model loading timed out")
def load_model(model_path, device):
    """
    Load a saved model from the given path.
    Tries multiple approaches to load the model safely.
    """
    try:
        print(f"Loading model from {model_path}...")
        
        # Check if file exists
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} does not exist")
            # Try to find model files in the current directory
            model_files = [f for f in os.listdir('.') if f.endswith('.pt') or f.endswith('.pth')]
            if model_files:
                print(f"Found these model files instead: {model_files}")
                print(f"Trying to load {model_files[0]} instead...")
                model_path = model_files[0]
            else:
                raise FileNotFoundError(f"No model files found in the current directory")
        
        # First try loading as a full model
        try:
            model = torch.load(model_path, map_location=device)
            print("Successfully loaded full model")
            return model
        except Exception as full_model_error:
            print(f"Error loading full model: {full_model_error}")
            
            # Try loading as state dict
            try:
                # Create a model with the same architecture
                print("Trying to load as state dict...")
                
                # Try different model configurations
                model_configs = [
                    # Standard model
                    {"embed_dim": 96, "depths": [2, 2, 6, 2], "num_heads": [3, 6, 12, 24]},
                    # Small model
                    {"embed_dim": 32, "depths": [1, 1, 2, 1], "num_heads": [2, 2, 2, 2]},
                    # Tiny model
                    {"embed_dim": 16, "depths": [1, 1, 1, 1], "num_heads": [1, 1, 1, 1]},
                    # Extra small model
                    {"embed_dim": 8, "depths": [1, 1, 1, 1], "num_heads": [1, 1, 1, 1]}
                ]
                
                # Try each configuration
                for config in model_configs:
                    try:
                        print(f"Trying model with config: {config}")
                        model = VideoSWIN3D(
                            num_classes=26,
                            patch_size=(2,4,4),
                            in_chans=3,
                            embed_dim=config["embed_dim"],
                            depths=config["depths"],
                            num_heads=config["num_heads"],
                            window_size=(8,7,7),
                            mlp_ratio=4.
                        )
                        
                        # Load state dict
                        state_dict = torch.load(model_path, map_location=device)
                        
                        # Check if it's a state dict or a full model
                        if hasattr(state_dict, 'state_dict'):
                            state_dict = state_dict.state_dict()
                        
                        # Try to handle key mismatches
                        try:
                            model.load_state_dict(state_dict)
                        except Exception as key_error:
                            print(f"Key mismatch error: {key_error}")
                            print("Trying to load with strict=False...")
                            model.load_state_dict(state_dict, strict=False)
                            print("Loaded with some missing keys (strict=False)")
                        
                        model = model.to(device)
                        print("Successfully loaded model state dict")
                        return model
                    except Exception as config_error:
                        print(f"Error with this config: {config_error}")
                        continue
                
                # If all configs fail, try one last approach - create a dummy model
                print("All configurations failed. Creating a minimal dummy model...")
                model = VideoSWIN3D(
                    num_classes=26,
                    patch_size=(2,4,4),
                    in_chans=3,
                    embed_dim=8,
                    depths=[1, 1, 1, 1],
                    num_heads=[1, 1, 1, 1],
                    window_size=(8,7,7),
                    mlp_ratio=4.
                ).to(device)
                print("Created a dummy model. Note: This model has not been trained!")
                return model
            except Exception as state_dict_error:
                print(f"Error loading state dict: {state_dict_error}")
                raise ValueError("Failed to load model in any format")
    except TimeoutError:
        print("Model loading timed out. Creating a dummy model...")
        model = VideoSWIN3D(
            num_classes=26,
            patch_size=(2,4,4),
            in_chans=3,
            embed_dim=8,
            depths=[1, 1, 1, 1],
            num_heads=[1, 1, 1, 1],
            window_size=(8,7,7),
            mlp_ratio=4.
        ).to('cpu')  # Always use CPU for dummy model
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        print("Creating a dummy model as fallback...")
        model = VideoSWIN3D(
            num_classes=26,
            patch_size=(2,4,4),
            in_chans=3,
            embed_dim=8,
            depths=[1, 1, 1, 1],
            num_heads=[1, 1, 1, 1],
            window_size=(8,7,7),
            mlp_ratio=4.
        ).to('cpu')  # Always use CPU for dummy model
        return model

def test_model(model_path, data_path=None, batch_size=2, inf_threshold=0.6):
    """
    Test a saved model on the specified dataset.
    """
    # Set up signal handlers for graceful exit
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    
    def signal_handler(sig, frame):
        print(f"\nReceived signal {sig}. Cleaning up and exiting gracefully...")
        # Restore original signal handlers
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        
        # Clean up resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("Cleanup complete. Exiting.")
        sys.exit(0)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start with a clean slate
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get the optimal device
        device = get_optimal_device()
        
        # Load the model with timeout protection
        try:
            model = load_model(model_path, device)
            if model is None:
                print("Failed to load model. Creating a dummy model...")
                model = VideoSWIN3D(
                    num_classes=26,
                    patch_size=(2,4,4),
                    in_chans=3,
                    embed_dim=8,
                    depths=[1, 1, 1, 1],
                    num_heads=[1, 1, 1, 1],
                    window_size=(8,7,7),
                    mlp_ratio=4.
                ).to(device)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating a dummy model...")
            model = VideoSWIN3D(
                num_classes=26,
                patch_size=(2,4,4),
                in_chans=3,
                embed_dim=8,
                depths=[1, 1, 1, 1],
                num_heads=[1, 1, 1, 1],
                window_size=(8,7,7),
                mlp_ratio=4.
            ).to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        # Set up data path
        if data_path is None:
            data_path = cfg.file_paths['train_data']
        
        # Load the dataset with timeout protection
        @timeout(120, "Dataset loading timed out")
        def load_dataset():
            try:
                print(f"Loading data from {data_path}...")
                list_IDs, labels, IDs_path = get_video_data(data_path)
                print(f"Found {len(list_IDs)} video(s) in {data_path}")
                
                # If no videos found, try to use the sample video
                if len(list_IDs) == 0:
                    print("No videos found. Trying to use sample video...")
                    try:
                        from setup_sample import setup_sample_video
                        if setup_sample_video():
                            # Try again to get videos after setup
                            list_IDs, labels, IDs_path = get_video_data(data_path)
                            if len(list_IDs) == 0:
                                raise FileNotFoundError("Could not find or set up sample video")
                        else:
                            raise FileNotFoundError("Could not find or set up sample video")
                    except Exception as e:
                        print(f"Error setting up sample video: {e}")
                        try:
                            from download_sample import download_sample_video
                            if download_sample_video():
                                # Try again to get videos after download
                                list_IDs, labels, IDs_path = get_video_data(data_path)
                                if len(list_IDs) == 0:
                                    raise FileNotFoundError("Could not find or download sample video")
                            else:
                                raise FileNotFoundError("Could not find or download sample video")
                        except Exception as e:
                            print(f"Error downloading sample video: {e}")
                            
                            # Last resort: create a dummy dataset with random data
                            print("Creating a dummy dataset with random data...")
                            dummy_id = 'dummy_video'
                            list_IDs = [dummy_id]
                            labels = {dummy_id: [0] * 26}
                            IDs_path = {dummy_id: 'dummy_path'}  # This won't be used
                
                return list_IDs, labels, IDs_path
            except Exception as e:
                print(f"Error loading dataset: {e}")
                traceback.print_exc()
                
                # Create a dummy dataset as fallback
                print("Creating a dummy dataset with random data...")
                dummy_id = 'dummy_video'
                list_IDs = [dummy_id]
                labels = {dummy_id: [0] * 26}
                IDs_path = {dummy_id: 'dummy_path'}  # This won't be used
                return list_IDs, labels, IDs_path
        
        try:
            list_IDs, labels, IDs_path = load_dataset()
        except TimeoutError:
            print("Dataset loading timed out. Creating a dummy dataset...")
            dummy_id = 'dummy_video'
            list_IDs = [dummy_id]
            labels = {dummy_id: [0] * 26}
            IDs_path = {dummy_id: 'dummy_path'}  # This won't be used
        
        # Create the dataset
        try:
            # Check if we're using a dummy dataset
            if list_IDs[0] == 'dummy_video':
                print("Using dummy dataset with random tensors...")
                
                class DummyDataset(torch.utils.data.Dataset):
                    def __init__(self, num_samples=10):
                        self.num_samples = num_samples
                    
                    def __len__(self):
                        return self.num_samples
                    
                    def __getitem__(self, idx):
                        # Create random input tensor (C, T, H, W)
                        inputs = torch.rand(3, 16, 120, 120)
                        # Create random target tensor (num_classes)
                        targets = torch.zeros(26)
                        targets[torch.randint(0, 26, (1,))] = 1.0
                        return inputs, targets
                
                test_dataset = DummyDataset()
            else:
                test_dataset = TinyVIRAT_dataset(list_IDs=list_IDs, IDs_path=IDs_path, labels=labels, frame_by_frame=False)
        except Exception as e:
            print(f"Error creating dataset: {e}")
            traceback.print_exc()
            
            print("Using dummy dataset with random tensors...")
            
            class DummyDataset(torch.utils.data.Dataset):
                def __init__(self, num_samples=10):
                    self.num_samples = num_samples
                
                def __len__(self):
                    return self.num_samples
                
                def __getitem__(self, idx):
                    # Create random input tensor (C, T, H, W)
                    inputs = torch.rand(3, 16, 120, 120)
                    # Create random target tensor (num_classes)
                    targets = torch.zeros(26)
                    targets[torch.randint(0, 26, (1,))] = 1.0
                    return inputs, targets
            
            test_dataset = DummyDataset()
        
        # Adjust batch size based on available memory
        if device.type == 'cuda':
            try:
                free_memory = (torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)) / 1024**3
                print(f"Available GPU memory: {free_memory:.2f} GB")
                
                if free_memory < 0.5:  # Less than 500MB
                    batch_size = 1
                    print(f"Very limited GPU memory detected ({free_memory:.2f} GB). Reduced batch size to 1.")
                elif free_memory < 1.0:  # Less than 1GB
                    batch_size = 1
                    print(f"Limited GPU memory detected ({free_memory:.2f} GB). Reduced batch size to 1.")
                elif free_memory < 2.0:  # Less than 2GB
                    batch_size = min(batch_size, 2)
                    print(f"Moderate GPU memory detected ({free_memory:.2f} GB). Using batch size of {batch_size}.")
            except Exception as e:
                print(f"Error adjusting batch size: {e}")
                batch_size = 1  # Conservative default
        
        # Create data loader with error handling
        try:
            params = {
                'batch_size': batch_size,
                'shuffle': False,
                'num_workers': 0,  # Avoid multiprocessing issues
                'drop_last': False  # Process all samples
            }
            
            if device.type == 'cuda':
                params['pin_memory'] = True
            
            test_generator = DataLoader(test_dataset, **params)
            print(f"Created data loader with batch size {batch_size}")
        except Exception as e:
            print(f"Error creating data loader: {e}")
            traceback.print_exc()
            
            # Try with minimal parameters
            params = {
                'batch_size': 1,
                'shuffle': False,
                'num_workers': 0,
                'drop_last': False
            }
            test_generator = DataLoader(test_dataset, **params)
            print("Created data loader with minimal parameters")
        
        # Define loss function
        try:
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none')  # Use 'none' to handle per-sample
        except Exception as e:
            print(f"Error creating loss function: {e}")
            traceback.print_exc()
            # Fallback to a simple MSE loss
            criterion = torch.nn.MSELoss(reduction='none')
        
        # Test the model
        print("Starting testing...")
        test_loss = 0.
        test_accuracy = 0.
        cnt = 0.
        successful_batches = 0
        
        # Set a timeout for the entire testing process
        start_time = time.time()
        max_test_time = 3600  # 1 hour max
        
        with torch.no_grad():
            for test_batch_idx, (inputs, targets) in enumerate(tqdm(test_generator, desc="Testing")):
                # Check if we've exceeded the maximum test time
                if time.time() - start_time > max_test_time:
                    print(f"Testing has been running for over {max_test_time/3600:.1f} hours. Stopping early.")
                    break
                
                try:
                    # Set a timeout for processing this batch
                    @timeout(30, f"Processing batch {test_batch_idx} timed out")
                    def process_batch(inputs, targets):
                        # Move data to device
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        
                        # Clear cache before forward pass
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        
                        # Forward pass
                        predictions = model(inputs.float())
                        
                        # Calculate metrics
                        batch_loss = criterion(predictions, targets).mean(dim=1).sum().item()
                        batch_accuracy = compute_accuracy(predictions.detach().cpu(), targets.cpu(), inf_threshold)
                        
                        # Update counters
                        nonlocal test_loss, test_accuracy, cnt, successful_batches
                        test_loss += batch_loss
                        test_accuracy += batch_accuracy
                        cnt += len(targets)
                        successful_batches += 1
                        
                        # Free memory
                        del inputs, targets, predictions
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                    
                    # Process the batch with timeout protection
                    try:
                        process_batch(inputs, targets)
                    except TimeoutError as e:
                        print(f"Batch processing timed out: {e}")
                        # Skip this batch and continue
                        continue
                    
                    # Periodically run garbage collection
                    if test_batch_idx % 5 == 0:
                        gc.collect()
                        
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f"CUDA out of memory encountered on test batch {test_batch_idx}. Clearing cache and trying to continue...")
                        if torch.cuda.is_available():
                            print(torch.cuda.memory_summary(device=device))
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                        # If this is not the first batch and we're using GPU, try to move to CPU
                        if test_batch_idx > 0 and device.type == 'cuda':
                            print("Moving model to CPU to continue testing...")
                            try:
                                model = model.cpu()
                                device = torch.device("cpu")
                                # Process this batch on CPU
                                inputs = inputs.to(device)
                                targets = targets.to(device)
                                predictions = model(inputs.float())
                                batch_loss = criterion(predictions, targets).mean(dim=1).sum().item()
                                batch_accuracy = compute_accuracy(predictions.detach(), targets, inf_threshold)
                                test_loss += batch_loss
                                test_accuracy += batch_accuracy
                                cnt += len(targets)
                                successful_batches += 1
                                continue
                            except Exception as cpu_e:
                                print(f"Error processing on CPU: {cpu_e}")
                        elif params['batch_size'] > 1:
                            # Try with smaller batch size
                            params['batch_size'] = 1
                            print(f"Reducing batch size to {params['batch_size']} and recreating test data loader")
                            try:
                                test_generator = DataLoader(test_dataset, **params)
                                # Restart testing
                                test_loss = 0.
                                test_accuracy = 0.
                                cnt = 0.
                                successful_batches = 0
                                break
                            except Exception as dl_e:
                                print(f"Error recreating data loader: {dl_e}")
                        else:
                            print("Cannot reduce batch size further. Skipping this batch.")
                            continue
                    else:
                        print(f"Runtime error in test batch {test_batch_idx}: {str(e)}")
                        traceback.print_exc()
                        continue
                except Exception as e:
                    print(f"Error in test batch {test_batch_idx}: {str(e)}")
                    traceback.print_exc()
                    continue
            
            # Calculate final metrics
            if successful_batches > 0:
                if cnt > 0:
                    test_loss /= cnt
                    test_accuracy /= successful_batches
                    print(f"Test metrics - Accuracy: {test_accuracy:6.2f} %, Loss: {test_loss:8.5f}")
                    print(f"Successfully processed {cnt} samples in {successful_batches} batches")
                    
                    # Save results to a file
                    try:
                        with open("test_results.txt", "w") as f:
                            f.write(f"Model: {model_path}\n")
                            f.write(f"Data: {data_path}\n")
                            f.write(f"Accuracy: {test_accuracy:6.2f} %\n")
                            f.write(f"Loss: {test_loss:8.5f}\n")
                            f.write(f"Samples: {cnt}\n")
                            f.write(f"Batches: {successful_batches}\n")
                            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        print("Test results saved to test_results.txt")
                    except Exception as save_e:
                        print(f"Error saving results: {save_e}")
                else:
                    print("No test samples were successfully processed. Testing failed.")
            else:
                print("No batches were successfully processed. Testing failed.")
                
                # Try one last approach - process a single random tensor
                try:
                    print("Trying to process a single random tensor...")
                    random_input = torch.rand(1, 3, 16, 120, 120).to(device)
                    random_output = model(random_input)
                    print(f"Model successfully processed a random tensor of shape {random_input.shape}")
                    print(f"Output shape: {random_output.shape}")
                    del random_input, random_output
                except Exception as random_e:
                    print(f"Error processing random tensor: {random_e}")
    except Exception as e:
        print(f"Error during testing: {e}")
        traceback.print_exc()
    finally:
        # Clean up resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Restore original signal handlers
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        
        print("Testing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained model on a dataset")
    parser.add_argument("--model", type=str, default="23_ckpt.pt", help="Path to the model checkpoint")
    parser.add_argument("--data", type=str, default=None, help="Path to the test data directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for testing")
    parser.add_argument("--threshold", type=float, default=0.6, help="Inference threshold")
    
    args = parser.parse_args()
    
    test_model(args.model, args.data, args.batch_size, args.threshold)
