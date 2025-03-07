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

# Force garbage collection to free up memory
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

import config as cfg

def compute_accuracy(pred, target, inf_th):
    target = target.cpu().data.numpy()
    pred = torch.sigmoid(pred)
    pred = pred.cpu().data.numpy()
    pred = pred > inf_th
    return accuracy_score(pred, target)

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
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                print(f"Using single available GPU: {torch.cuda.get_device_name(0)}")
                return torch.device("cuda:0")
            except RuntimeError as e:
                print(f"Error with GPU: {e}")
                print("Falling back to CPU...")
                return torch.device("cpu")
        
        # Multiple GPUs available, find the one with most free memory
        free_memory = []
        for i in range(num_gpus):
            try:
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                free_memory.append(torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i))
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
        print("Falling back to CPU...")
        return torch.device("cpu")

def load_model(model_path, device):
    """
    Load a saved model from the given path.
    Tries multiple approaches to load the model safely.
    """
    try:
        print(f"Loading model from {model_path}...")
        
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
                    {"embed_dim": 16, "depths": [1, 1, 1, 1], "num_heads": [1, 1, 1, 1]}
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
                        model.load_state_dict(state_dict)
                        model = model.to(device)
                        print("Successfully loaded model state dict")
                        return model
                    except Exception as config_error:
                        print(f"Error with this config: {config_error}")
                        continue
                
                # If all configs fail, raise an error
                raise ValueError("Could not load model with any configuration")
            except Exception as state_dict_error:
                print(f"Error loading state dict: {state_dict_error}")
                raise ValueError("Failed to load model in any format")
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return None

def test_model(model_path, data_path=None, batch_size=2, inf_threshold=0.6):
    """
    Test a saved model on the specified dataset.
    """
    try:
        # Get the optimal device
        device = get_optimal_device()
        
        # Load the model
        model = load_model(model_path, device)
        if model is None:
            print("Failed to load model. Exiting.")
            return
        
        # Set model to evaluation mode
        model.eval()
        
        # Set up data path
        if data_path is None:
            data_path = cfg.file_paths['train_data']
        
        # Load the dataset
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
                        raise
            
            # Create the dataset
            test_dataset = TinyVIRAT_dataset(list_IDs=list_IDs, IDs_path=IDs_path, labels=labels, frame_by_frame=False)
            
            # Adjust batch size based on available memory
            if device.type == 'cuda':
                try:
                    free_memory = (torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)) / 1024**3
                    print(f"Available GPU memory: {free_memory:.2f} GB")
                    
                    if free_memory < 1.0:  # Less than 1GB
                        batch_size = 1
                        print(f"Limited GPU memory detected ({free_memory:.2f} GB). Reduced batch size to 1.")
                except Exception as e:
                    print(f"Error adjusting batch size: {e}")
            
            # Create data loader
            params = {
                'batch_size': batch_size,
                'shuffle': False,
                'num_workers': 0
            }
            
            if device.type == 'cuda':
                params['pin_memory'] = True
            
            test_generator = DataLoader(test_dataset, **params)
            
            # Define loss function
            criterion = torch.nn.BCEWithLogitsLoss()
            
            # Test the model
            print("Starting testing...")
            test_loss = 0.
            test_accuracy = 0.
            cnt = 0.
            
            with torch.no_grad():
                for test_batch_idx, (inputs, targets) in enumerate(tqdm(test_generator, desc="Testing")):
                    try:
                        # Move data to device
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        
                        # Clear cache before forward pass
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        
                        # Forward pass
                        predictions = model(inputs.float())
                        
                        # Calculate metrics
                        batch_loss = criterion(predictions, targets).sum().item()
                        batch_accuracy = compute_accuracy(predictions.detach().cpu(), targets.cpu(), inf_threshold)
                        
                        # Update counters
                        test_loss += batch_loss
                        test_accuracy += batch_accuracy
                        cnt += len(targets)
                        
                        # Free memory
                        del inputs, targets, predictions
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        
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
                                model = model.cpu()
                                device = torch.device("cpu")
                                continue
                            elif params['batch_size'] > 1:
                                # Try with smaller batch size
                                params['batch_size'] = 1
                                print(f"Reducing batch size to {params['batch_size']} and recreating test data loader")
                                test_generator = DataLoader(test_dataset, **params)
                                # Restart testing
                                test_loss = 0.
                                test_accuracy = 0.
                                cnt = 0.
                                break
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
                if cnt > 0:
                    test_loss /= cnt
                    test_accuracy /= (test_batch_idx + 1)
                    print(f"Test metrics - Accuracy: {test_accuracy:6.2f} %, Loss: {test_loss:8.5f}")
                else:
                    print("No test samples were successfully processed. Testing failed.")
        except Exception as data_e:
            print(f"Error loading or processing data: {data_e}")
            traceback.print_exc()
    except Exception as e:
        print(f"Error during testing: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained model on a dataset")
    parser.add_argument("--model", type=str, default="23_ckpt.pt", help="Path to the model checkpoint")
    parser.add_argument("--data", type=str, default=None, help="Path to the test data directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for testing")
    parser.add_argument("--threshold", type=float, default=0.6, help="Inference threshold")
    
    args = parser.parse_args()
    
    test_model(args.model, args.data, args.batch_size, args.threshold)