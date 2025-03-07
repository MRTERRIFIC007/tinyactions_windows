from matplotlib.pyplot import get
import torch
import numpy as np
from Model.VideoSWIN import VideoSWIN3D
from my_dataloader import TinyVIRAT_dataset, get_video_data
from asam import ASAM
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.visualize import get_plot
from sklearn.metrics import accuracy_score
import os
import multiprocessing
import traceback
import gc

# Force garbage collection to free up memory
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

import config as cfg

exp = '23'

# Make exp dir
if not os.path.exists('exps/exp_' + exp + '/'):
    os.makedirs('exps/exp_' + exp + '/')
PATH = 'exps/exp_' + exp + '/'

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

def main():
    try:
        # Training Parameters
        shuffle = True
        print("Creating params....")
        params = {'batch_size': 2,
                  'shuffle': shuffle,
                  'num_workers': 0}  # Set to 0 to avoid multiprocessing issues

        max_epochs = 3
        inf_threshold = 0.6
        print(params)
    except Exception as e:
        print("Error during training parameter initialization:", str(e))
        traceback.print_exc()
        return

    # Get the optimal device
    device = get_optimal_device()
    
    # Adjust batch size based on available memory
    if device.type == 'cuda':
        try:
            # Get available GPU memory in GB
            free_memory = (torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)) / 1024**3
            print(f"Available GPU memory: {free_memory:.2f} GB")
            
            # Adjust batch size based on available memory
            if free_memory < 1.0:  # Less than 1GB
                params['batch_size'] = 1
                print(f"Limited GPU memory detected ({free_memory:.2f} GB). Reduced batch size to 1.")
            elif free_memory < 2.0:  # Less than 2GB
                params['batch_size'] = 2
                print(f"Moderate GPU memory detected ({free_memory:.2f} GB). Using batch size of 2.")
            else:  # More than 2GB
                params['batch_size'] = 4
                print(f"Sufficient GPU memory detected ({free_memory:.2f} GB). Increased batch size to 4.")
        except Exception as e:
            print(f"Error adjusting batch size: {e}")
            # Keep default batch size
        
        # Enable pin_memory for faster data transfer to GPU
        params['pin_memory'] = True
        print("pin_memory enabled for DataLoader")

    ############ Data Generators ############
    # CHANGE THIS PATH: Update the following path to the location of your training videos on your system
    try:
        video_root = cfg.file_paths['train_data']
        # Recursively collect video file paths and dummy labels from the folder
        list_IDs, labels, IDs_path = get_video_data(video_root)
        print(f"Found {len(list_IDs)} video(s) in {video_root}")
        
        # If no videos found in the dataset directory, use the sample video
        if len(list_IDs) == 0:
            print("No videos found in the dataset directory. Using the sample video.mp4 instead.")
            
            # Try multiple possible locations for the sample video
            script_dir = os.path.dirname(os.path.abspath(__file__))
            cwd = os.getcwd()
            possible_paths = [
                os.path.join(script_dir, 'video.mp4'),  # In the same directory as train.py
                os.path.join(script_dir, '..', 'video.mp4'),  # One level up
                os.path.join(script_dir, '..', '..', 'video.mp4'),  # Two levels up
                os.path.join(cwd, 'video.mp4'),  # In the current working directory
                '/workspace/video.mp4',  # Root of workspace (common in Docker/Jupyter)
                '/workspace/Tinyactions_2/video.mp4',  # Common Docker/Jupyter path
                '/workspace/Tinyactions_2/tinyactions_windows/video.mp4',  # Specific to your environment
            ]
            
            sample_video_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    sample_video_path = path
                    break
            
            if sample_video_path:
                # Create a single video dataset with dummy labels
                dummy_list_IDs = ['sample_video']
                dummy_labels = {'sample_video': [0] * cfg.constants['num_classes']}
                dummy_IDs_path = {'sample_video': sample_video_path}
                list_IDs, labels, IDs_path = dummy_list_IDs, dummy_labels, dummy_IDs_path
                print(f"Using sample video: {sample_video_path}")
            else:
                # If sample video not found, try to download one
                try:
                    print("Sample video not found. Attempting to download a sample video...")
                    from download_sample import download_sample_video
                    if download_sample_video():
                        # Try again to get videos after download
                        list_IDs, labels, IDs_path = get_video_data(video_root)
                        if len(list_IDs) == 0:
                            raise FileNotFoundError("Could not find or download sample video")
                    else:
                        # If download fails, try the setup script
                        from setup_sample import setup_sample_video
                        if setup_sample_video():
                            # Try again to get videos after setup
                            list_IDs, labels, IDs_path = get_video_data(video_root)
                            if len(list_IDs) == 0:
                                raise FileNotFoundError("Could not find, download, or set up sample video")
                        else:
                            raise FileNotFoundError("Could not find, download, or set up sample video")
                except Exception as e:
                    print(f"Error setting up sample video: {e}")
                    raise

        # Create the dataset using TinyVIRAT_dataset
        train_dataset = TinyVIRAT_dataset(list_IDs=list_IDs, IDs_path=IDs_path, labels=labels, frame_by_frame=False)
        training_generator = DataLoader(train_dataset, **params)
    except Exception as e:
        print("Error while creating data generator:", str(e))
        traceback.print_exc()
        return

    # Optionally, you may want to also override the validation set or remove it entirely,
    # depending on your experiment setup.

    try:
        # Choose model size based on available memory
        if device.type == 'cuda':
            try:
                # Get available GPU memory in GB
                free_memory = (torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)) / 1024**3
                
                # Select model size based on available memory
                if free_memory < 1.0:  # Less than 1GB
                    print(f"Limited GPU memory detected ({free_memory:.2f} GB). Using tiny model.")
                    model = VideoSWIN3D(
                        num_classes=26,
                        patch_size=(2,4,4),
                        in_chans=3,
                        embed_dim=16,    # minimal embedding dimension
                        depths=[1, 1, 1, 1],    # minimal depth
                        num_heads=[1, 1, 1, 1],  # minimal heads
                        window_size=(8,7,7),
                        mlp_ratio=4.
                    )
                elif free_memory < 2.0:  # Less than 2GB
                    print(f"Moderate GPU memory detected ({free_memory:.2f} GB). Using small model.")
                    model = VideoSWIN3D(
                        num_classes=26,
                        patch_size=(2,4,4),
                        in_chans=3,
                        embed_dim=32,    # reduced embedding dimension
                        depths=[1, 1, 2, 1],    # shallower network
                        num_heads=[2, 2, 2, 2],  # fewer attention heads
                        window_size=(8,7,7),
                        mlp_ratio=4.
                    )
                else:  # More than 2GB
                    print(f"Sufficient GPU memory detected ({free_memory:.2f} GB). Using standard model.")
                    model = VideoSWIN3D(
                        num_classes=26,
                        patch_size=(2,4,4),
                        in_chans=3,
                        embed_dim=96,
                        depths=[2, 2, 6, 2],
                        num_heads=[3, 6, 12, 24],
                        window_size=(8,7,7),
                        mlp_ratio=4.
                    )
            except Exception as e:
                print(f"Error selecting model size: {e}")
                # Fall back to small model
                model = VideoSWIN3D(
                    num_classes=26,
                    patch_size=(2,4,4),
                    in_chans=3,
                    embed_dim=32,
                    depths=[1, 1, 2, 1],
                    num_heads=[2, 2, 2, 2],
                    window_size=(8,7,7),
                    mlp_ratio=4.
                )
        else:
            # For CPU, use the smallest model
            print("Using CPU. Selecting tiny model for better performance.")
            model = VideoSWIN3D(
                num_classes=26,
                patch_size=(2,4,4),
                in_chans=3,
                embed_dim=16,
                depths=[1, 1, 1, 1],
                num_heads=[1, 1, 1, 1],
                window_size=(8,7,7),
                mlp_ratio=4.
            )
        
        # Try to move model to the selected device, fall back to CPU if there's an error
        try:
            model = model.to(device)
        except RuntimeError as cuda_err:
            if device.type == 'cuda':
                print(f"Error moving model to CUDA: {cuda_err}")
                print("Falling back to CPU...")
                device = torch.device("cpu")
                # Use the smallest model for CPU
                model = VideoSWIN3D(
                    num_classes=26,
                    patch_size=(2,4,4),
                    in_chans=3,
                    embed_dim=16,
                    depths=[1, 1, 1, 1],
                    num_heads=[1, 1, 1, 1],
                    window_size=(8,7,7),
                    mlp_ratio=4.
                ).to(device)
            else:
                raise  # Re-raise if it's not a CUDA device
    except Exception as e:
        print("Error while initializing the model:", e)
        traceback.print_exc()
        print("Attempting to initialize a smaller model due to potential GPU memory issues...")
        try:
            model = VideoSWIN3D(
                num_classes=26,
                patch_size=(2,4,4),
                in_chans=3,
                embed_dim=32,    # reduced embedding dimension for a smaller model
                depths=[1, 1, 2, 1],    # shallower network
                num_heads=[2, 2, 2, 2],  # fewer attention heads
                window_size=(8,7,7),
                mlp_ratio=4.
            )
            
            # Try to move model to the selected device, fall back to CPU if there's an error
            try:
                model = model.to(device)
            except RuntimeError as cuda_err:
                if device.type == 'cuda':
                    print(f"Error moving smaller model to CUDA: {cuda_err}")
                    print("Falling back to CPU...")
                    device = torch.device("cpu")
                    model = model.to(device)
                else:
                    raise  # Re-raise if it's not a CUDA device
        except Exception as e2:
            print(f"Error initializing smaller model: {e2}")
            print("Falling back to CPU with minimal model...")
            device = torch.device("cpu")
            model = VideoSWIN3D(
                num_classes=26,
                patch_size=(2,4,4),
                in_chans=3,
                embed_dim=16,    # minimal embedding dimension
                depths=[1, 1, 1, 1],    # minimal depth
                num_heads=[1, 1, 1, 1],  # minimal heads
                window_size=(8,7,7),
                mlp_ratio=4.
            ).to(device)

    try:
        # Load pre-trained weights if available
        pretrained_path = '/path/to/your/pretrained/weights.pth'
        if os.path.exists(pretrained_path):
            model.load_pretrained_weights(pretrained_path)
        else:
            print("No pre-trained weights found. Training from scratch.")
    except Exception as e:
        print("Error while loading pre-trained weights:", str(e))
        traceback.print_exc()
        # Not fatal; continuing...

    try:
        # Define loss and optimizer
        lr = 0.02
        wt_decay = 5e-4
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Adjust learning rate based on device and model size
        if device.type == 'cpu':
            # Lower learning rate for CPU training
            lr = 0.005
            print(f"Using reduced learning rate for CPU: {lr}")
        elif hasattr(model, 'embed_dim') and model.embed_dim < 96:
            # Lower learning rate for smaller models
            lr = 0.01
            print(f"Using reduced learning rate for small model: {lr}")
        
        # Use Adam optimizer for better memory efficiency on small GPUs
        if device.type == 'cuda' and (torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)) / 1024**3 < 2.0:
            print("Using Adam optimizer for better memory efficiency")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wt_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wt_decay)

        # ASAM with adjusted parameters for memory efficiency
        if device.type == 'cuda' and (torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)) / 1024**3 < 1.0:
            # Use smaller rho for limited memory
            rho = 0.1
            eta = 0.01
            print(f"Using reduced ASAM parameters for limited memory: rho={rho}")
        else:
            rho = 0.55
            eta = 0.01
        
        minimizer = ASAM(optimizer, model, rho=rho, eta=eta)
    except Exception as e:
        print("Error during loss, optimizer, or ASAM initialization:", str(e))
        traceback.print_exc()
        return

    # Training and validation loops
    epoch_loss_train = []
    epoch_acc_train = []

    best_accuracy = 0.
    print("Begin Training....")
    for epoch in range(max_epochs):
        try:
            # Train
            model.train()
            loss = 0.
            accuracy = 0.
            cnt = 0.

            for batch_idx, (inputs, targets) in enumerate(tqdm(training_generator)):
                try:
                    # Move data to device
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    # Clear gradients
                    optimizer.zero_grad()

                    # Free up memory before forward pass
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # Use gradient checkpointing to save memory if available
                    if hasattr(model, 'set_grad_checkpointing'):
                        model.set_grad_checkpointing(True)

                    # Ascent Step
                    predictions = model(inputs.float())
                    batch_loss = criterion(predictions, targets)
                    batch_loss.mean().backward()
                    minimizer.ascent_step()

                    # Free memory after backward pass
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                    # Descent Step
                    descent_loss = criterion(model(inputs.float()), targets)
                    descent_loss.mean().backward()
                    minimizer.descent_step()

                    # Calculate metrics on CPU to save GPU memory
                    with torch.no_grad():
                        loss += batch_loss.sum().item()
                        # Move predictions to CPU before computing accuracy
                        accuracy += compute_accuracy(predictions.detach().cpu(), targets.cpu(), inf_threshold)
                    cnt += len(targets)
                    
                    # Explicitly delete tensors to free memory
                    del inputs, targets, predictions, batch_loss, descent_loss
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # Periodically run garbage collection
                    if batch_idx % 10 == 0:
                        gc.collect()
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f"CUDA out of memory encountered on batch {batch_idx} at epoch {epoch}. Clearing cache, collecting garbage, and printing memory summary.")
                        if torch.cuda.is_available():
                            print(torch.cuda.memory_summary(device=device))
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                        # Try to reduce batch size dynamically
                        if params['batch_size'] > 1:
                            params['batch_size'] = params['batch_size'] // 2
                            print(f"Reducing batch size to {params['batch_size']} and recreating data loader")
                            # Recreate data loader with smaller batch size
                            training_generator = DataLoader(train_dataset, **params)
                            # Skip to next epoch
                            break
                        continue
                    else:
                        print(f"Runtime error processing batch {batch_idx} in epoch {epoch}: {str(e)}")
                        traceback.print_exc()
                        continue
                except Exception as batch_e:
                    print(f"Error processing batch {batch_idx} in epoch {epoch}: {str(batch_e)}")
                    traceback.print_exc()
                    continue

            loss /= cnt
            accuracy /= (batch_idx + 1)
            print(f"Epoch: {epoch}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}")
            epoch_loss_train.append(loss)
            epoch_acc_train.append(accuracy)

            # Save best model based on training accuracy
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), PATH + exp + '_best_ckpt.pt')
        except Exception as epoch_e:
            print(f"Error during epoch {epoch}: {str(epoch_e)}")
            traceback.print_exc()
            continue   # Skip the epoch and continue to the next one

    print(f"Best train accuracy: {best_accuracy}")
    print("TRAINING COMPLETED :)")

    # Testing phase on the entire training dataset
    try:
        print("Testing on the entire training dataset...")
        
        # Make sure we have enough memory for testing
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
            
            # Check if we have enough memory for testing
            try:
                free_memory = (torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)) / 1024**3
                print(f"Available GPU memory for testing: {free_memory:.2f} GB")
                
                # If memory is very low, consider moving to CPU for testing
                if free_memory < 0.5:  # Less than 500MB
                    print("Very limited GPU memory available. Consider testing on CPU...")
                    use_cpu_for_testing = input("Use CPU for testing instead of GPU? (y/n): ").lower() == 'y'
                    if use_cpu_for_testing:
                        print("Moving model to CPU for testing...")
                        model = model.cpu()
                        device = torch.device("cpu")
            except Exception as e:
                print(f"Error checking GPU memory: {e}")
        
        # Set model to evaluation mode
        model.eval()
        test_loss = 0.
        test_accuracy = 0.
        cnt = 0.
        
        # Create a smaller batch size for testing if needed
        test_params = params.copy()
        if device.type == 'cuda' and params['batch_size'] > 1:
            try:
                free_memory = (torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)) / 1024**3
                if free_memory < 1.0:
                    test_params['batch_size'] = 1
                    print(f"Reduced batch size to {test_params['batch_size']} for testing due to limited memory")
            except Exception as e:
                print(f"Error adjusting test batch size: {e}")
        
        # Create a separate test data loader if needed
        try:
            if test_params != params:
                test_generator = DataLoader(train_dataset, **test_params)
                print(f"Created separate test data loader with batch size {test_params['batch_size']}")
            else:
                test_generator = training_generator
        except Exception as e:
            print(f"Error creating test data loader: {e}")
            print("Falling back to training generator")
            test_generator = training_generator
        
        # Perform testing with no gradient computation
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
                        elif test_params['batch_size'] > 1:
                            # Try with smaller batch size
                            test_params['batch_size'] = 1
                            print(f"Reducing batch size to {test_params['batch_size']} and recreating test data loader")
                            test_generator = DataLoader(train_dataset, **test_params)
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
    except Exception as test_e:
        print("Error during testing:", str(test_e))
        traceback.print_exc()
        print("Testing failed, but training was completed successfully.")

    try:
        # Save visualization
        get_plot(PATH, epoch_acc_train, None, 'Accuracy-' + exp, 'Train Accuracy', 'Val Accuracy (N/A)', 'Epochs', 'Acc')
        get_plot(PATH, epoch_loss_train, None, 'Loss-' + exp, 'Train Loss', 'Val Loss (N/A)', 'Epochs', 'Loss')
        print("Successfully saved visualization plots.")
    except Exception as e:
        print("Error while plotting:", str(e))
        traceback.print_exc()
        print("Continuing without saving plots...")

    try:
        # Save trained model
        print("Saving final model checkpoint...")
        
        # First try to save just the state dict (smaller file)
        try:
            save_path = exp + "_ckpt.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Successfully saved model state dict to {save_path}")
        except Exception as state_dict_error:
            print(f"Error saving model state dict: {state_dict_error}")
            
            # If that fails, try saving to a different location
            try:
                alternate_path = os.path.join(PATH, exp + "_ckpt.pt")
                torch.save(model.state_dict(), alternate_path)
                print(f"Successfully saved model state dict to alternate path: {alternate_path}")
            except Exception as alt_path_error:
                print(f"Error saving to alternate path: {alt_path_error}")
                
                # If that also fails, try saving the full model
                try:
                    print("Attempting to save full model instead...")
                    torch.save(model, PATH + exp + "_full_model.pt")
                    print(f"Successfully saved full model to {PATH + exp + '_full_model.pt'}")
                except Exception as full_model_error:
                    print(f"Error saving full model: {full_model_error}")
                    
                    # Last resort: try saving to CPU first
                    try:
                        print("Moving model to CPU and trying to save...")
                        cpu_model = model.cpu()
                        torch.save(cpu_model.state_dict(), PATH + exp + "_cpu_ckpt.pt")
                        print(f"Successfully saved CPU model to {PATH + exp + '_cpu_ckpt.pt'}")
                        # Move model back to original device
                        model = model.to(device)
                    except Exception as cpu_error:
                        print(f"All attempts to save model failed: {cpu_error}")
                        print("WARNING: Model was not saved!")
    except Exception as e:
        print("Error while saving the trained model:", str(e))
        traceback.print_exc()
        print("WARNING: Model was not saved!")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
