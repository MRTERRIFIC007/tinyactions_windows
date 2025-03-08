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
import time
import random

# Import for mixed precision training
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    print("Automatic Mixed Precision not available")

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
        
        # Determine optimal batch size based on available resources
        if torch.cuda.is_available():
            try:
                free_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"Total GPU memory: {free_memory:.2f} GB")
                if free_memory > 8:
                    batch_size = 8
                elif free_memory > 4:
                    batch_size = 4
                elif free_memory > 2:
                    batch_size = 2
                else:
                    batch_size = 1
            except:
                batch_size = 2  # Default if can't determine memory
        else:
            # For CPU, use a smaller batch size but with gradient accumulation
            batch_size = 1
        
        # Set up gradient accumulation steps (process multiple batches before updating weights)
        # This helps with small batch sizes
        if batch_size < 4:
            grad_accumulation_steps = 4 // batch_size
            print(f"Using gradient accumulation with {grad_accumulation_steps} steps")
        else:
            grad_accumulation_steps = 1
        
        # Increase epochs for CPU training to compensate for smaller model
        if not torch.cuda.is_available():
            max_epochs = 20  # More epochs for CPU training (increased from 10)
        else:
            # Increase GPU epochs as well for better accuracy
            if torch.cuda.get_device_properties(0).total_memory / 1024**3 < 4:
                max_epochs = 10  # For GPUs with less than 4GB memory
            else:
                max_epochs = 15  # For GPUs with more memory
        
        # Add early stopping to prevent overfitting
        early_stopping_patience = 5
        early_stopping_counter = 0
        best_val_metric = 0
        
        params = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': 0  # Set to 0 to avoid multiprocessing issues
        }
        
        inf_threshold = 0.6
        print(f"Training parameters: batch_size={batch_size}, grad_accumulation={grad_accumulation_steps}, epochs={max_epochs}, early_stopping_patience={early_stopping_patience}")
    except Exception as e:
        print("Error during training parameter initialization:", str(e))
        traceback.print_exc()
        return

    # Get the optimal device
    device = get_optimal_device()
    
    # Set up mixed precision training if available
    use_amp = AMP_AVAILABLE and device.type == 'cuda'
    if use_amp:
        print("Using Automatic Mixed Precision training")
        scaler = GradScaler()
    else:
        print("Mixed precision not available, using full precision")
    
    # If using CUDA, enable pin_memory in DataLoader for faster data transfer
    if device.type == 'cuda':
        params['pin_memory'] = True
        print("pin_memory enabled for DataLoader")

    ############ Data Generators ############
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

        # Create the dataset using TinyVIRAT_dataset with augmentation
        train_dataset = TinyVIRAT_dataset(
            list_IDs=list_IDs, 
            IDs_path=IDs_path, 
            labels=labels, 
            frame_by_frame=False,
            use_augmentation=True,
            is_training=True
        )
        training_generator = DataLoader(train_dataset, **params)
        
        # Create a validation set (20% of training data)
        if len(list_IDs) > 5:  # Only if we have enough data
            val_size = max(1, int(len(list_IDs) * 0.2))
            train_size = len(list_IDs) - val_size
            
            # Split the data
            train_list_IDs = list_IDs[:train_size]
            val_list_IDs = list_IDs[train_size:]
            
            # Create validation dataset and dataloader (no augmentation for validation)
            val_dataset = TinyVIRAT_dataset(
                list_IDs=[id for id in val_list_IDs], 
                IDs_path=IDs_path, 
                labels=labels, 
                frame_by_frame=False,
                use_augmentation=False,
                is_training=False
            )
            val_params = params.copy()
            val_params['shuffle'] = False  # No need to shuffle validation data
            val_generator = DataLoader(val_dataset, **val_params)
            print(f"Created validation set with {len(val_list_IDs)} videos")
            
            # Update training dataset
            train_dataset = TinyVIRAT_dataset(
                list_IDs=[id for id in train_list_IDs], 
                IDs_path=IDs_path, 
                labels=labels, 
                frame_by_frame=False,
                use_augmentation=True,
                is_training=True
            )
            training_generator = DataLoader(train_dataset, **params)
            print(f"Updated training set with {len(train_list_IDs)} videos")
            
            has_validation = True
        else:
            # Not enough data for validation, use training data for validation
            val_generator = training_generator
            has_validation = False
            print("Not enough data for separate validation set, using training data for validation")
    except Exception as e:
        print("Error while creating data generator:", str(e))
        traceback.print_exc()
        return

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
            # For CPU, use the smallest model but with more width
            print("Using CPU. Selecting optimized model for CPU training.")
            model = VideoSWIN3D(
                num_classes=26,
                patch_size=(2,4,4),
                in_chans=3,
                embed_dim=32,  # Increased from 24 for better representation
                depths=[1, 1, 2, 1],  # Slightly deeper model
                num_heads=[2, 2, 2, 2],  # More heads for better representation
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
        
        # Use focal loss for better handling of class imbalance
        try:
            from torch.nn import functional as F
            
            class FocalLoss(torch.nn.Module):
                def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
                    super(FocalLoss, self).__init__()
                    self.gamma = gamma
                    self.alpha = alpha
                    self.reduction = reduction
                
                def forward(self, inputs, targets):
                    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
                    pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
                    F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
                    
                    if self.reduction == 'sum':
                        return F_loss.sum()
                    elif self.reduction == 'mean':
                        return F_loss.mean()
                    else:
                        return F_loss
            
            # Use focal loss for better handling of imbalanced data
            criterion = FocalLoss(gamma=2.0, alpha=0.25)
            print("Using Focal Loss for better handling of class imbalance")
        except Exception as e:
            print(f"Could not initialize Focal Loss: {e}. Using BCEWithLogitsLoss instead.")
            criterion = torch.nn.BCEWithLogitsLoss()
        
        # Adjust learning rate based on device and model size
        if device.type == 'cpu':
            # Lower learning rate for CPU training
            lr = 0.003
            print(f"Using reduced learning rate for CPU: {lr}")
        elif hasattr(model, 'embed_dim') and model.embed_dim < 96:
            # Lower learning rate for smaller models
            lr = 0.008
            print(f"Using reduced learning rate for small model: {lr}")
        
        # Use cosine annealing with warm restarts for better convergence
        try:
            # Use AdamW optimizer for better generalization
            if device.type == 'cuda' and (torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)) / 1024**3 < 2.0:
                print("Using AdamW optimizer for better memory efficiency and generalization")
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wt_decay)
            else:
                print("Using SGD optimizer with momentum")
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wt_decay)
            
            # Use cosine annealing with warm restarts
            T_0 = max(5, max_epochs // 3)  # Restart every T_0 epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, T_mult=1, eta_min=lr/10
            )
            print(f"Using CosineAnnealingWarmRestarts scheduler with T_0={T_0}")
            
            # Create a backup scheduler for plateau detection
            plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=3, verbose=True
            )
            use_plateau_scheduler = False  # Start with cosine, switch to plateau if needed
        except Exception as e:
            print(f"Error setting up advanced scheduler: {e}")
            # Fallback to simple scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=2, verbose=True
            )
            use_plateau_scheduler = True

        # ASAM with adjusted parameters for memory efficiency
        if device.type == 'cuda' and (torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)) / 1024**3 < 1.0:
            # Use smaller rho for limited memory
            rho = 0.1
            eta = 0.01
            print(f"Using reduced ASAM parameters for limited memory: rho={rho}")
        else:
            rho = 0.55
            eta = 0.01
        
        # Initialize ASAM minimizer
        try:
            minimizer = ASAM(optimizer, model, rho=rho, eta=eta)
            use_asam = True
            print(f"Using ASAM optimizer with rho={rho}, eta={eta}")
        except Exception as e:
            print(f"Error initializing ASAM: {e}. Using standard optimizer.")
            use_asam = False
    except Exception as e:
        print("Error during loss, optimizer, or ASAM initialization:", str(e))
        traceback.print_exc()
        return

    # Training and validation loops
    epoch_loss_train = []
    epoch_acc_train = []
    epoch_loss_val = []
    epoch_acc_val = []

    best_accuracy = 0.
    best_val_accuracy = 0.
    print("Begin Training....")
    for epoch in range(max_epochs):
        try:
            # Train
            model.train()
            loss = 0.
            accuracy = 0.
            cnt = 0.
            batch_count = 0

            for batch_idx, (inputs, targets) in enumerate(tqdm(training_generator)):
                try:
                    # Move data to device
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    # Apply mixup data augmentation (with 50% probability)
                    apply_mixup = random.random() < 0.5 and len(inputs) > 1
                    if apply_mixup:
                        try:
                            # Create mixed samples
                            alpha = 0.2  # Mixup interpolation strength
                            lam = np.random.beta(alpha, alpha)
                            batch_size = inputs.size(0)
                            index = torch.randperm(batch_size).to(device)
                            
                            # Mix inputs and targets
                            mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                            mixed_targets = lam * targets + (1 - lam) * targets[index]
                            
                            # Use mixed data
                            inputs = mixed_inputs
                            targets = mixed_targets
                        except Exception as mixup_e:
                            print(f"Error applying mixup: {mixup_e}")
                            # Continue with original data
                            apply_mixup = False

                    # Only zero gradients at the beginning of accumulation steps
                    if batch_idx % grad_accumulation_steps == 0:
                        optimizer.zero_grad()

                    # Free up memory before forward pass
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # Use gradient checkpointing to save memory if available
                    if hasattr(model, 'set_grad_checkpointing'):
                        model.set_grad_checkpointing(True)

                    # Forward pass with mixed precision if available
                    if use_amp:
                        with autocast():
                            # Forward pass
                            predictions = model(inputs.float())
                            batch_loss = criterion(predictions, targets)
                            # Scale loss by accumulation steps
                            batch_loss = batch_loss.mean() / grad_accumulation_steps
                            
                        # Backward pass with gradient scaling
                        scaler.scale(batch_loss).backward()
                        
                        # Only update weights after accumulating gradients
                        if (batch_idx + 1) % grad_accumulation_steps == 0 or (batch_idx + 1) == len(training_generator):
                            if use_asam:
                                # Ascent step with ASAM
                                scaler.unscale_(optimizer)
                                minimizer.ascent_step()
                                
                                # Descent step
                                with autocast():
                                    descent_predictions = model(inputs.float())
                                    descent_loss = criterion(descent_predictions, targets)
                                    descent_loss = descent_loss.mean() / grad_accumulation_steps
                                
                                scaler.scale(descent_loss).backward()
                                scaler.unscale_(optimizer)
                                minimizer.descent_step()
                            else:
                                # Standard optimizer step
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                scaler.step(optimizer)
                            
                            # Update scaler
                            scaler.update()
                            
                            # Zero gradients after update
                            optimizer.zero_grad()
                    else:
                        # Standard precision training
                        # Forward pass
                        predictions = model(inputs.float())
                        batch_loss = criterion(predictions, targets)
                        # Scale loss by accumulation steps
                        batch_loss = batch_loss.mean() / grad_accumulation_steps
                        batch_loss.backward()
                        
                        # Only update weights after accumulating gradients
                        if (batch_idx + 1) % grad_accumulation_steps == 0 or (batch_idx + 1) == len(training_generator):
                            if use_asam:
                                # ASAM optimizer steps
                                minimizer.ascent_step()
                                
                                # Descent Step
                                descent_predictions = model(inputs.float())
                                descent_loss = criterion(descent_predictions, targets)
                                descent_loss = descent_loss.mean() / grad_accumulation_steps
                                descent_loss.backward()
                                minimizer.descent_step()
                            else:
                                # Standard optimizer step
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                optimizer.step()
                            
                            # Zero gradients after update
                            optimizer.zero_grad()

                    # Calculate metrics on CPU to save GPU memory
                    with torch.no_grad():
                        loss += batch_loss.sum().item() * grad_accumulation_steps
                        # Move predictions to CPU before computing accuracy
                        accuracy += compute_accuracy(predictions.detach().cpu(), targets.cpu(), inf_threshold)
                    cnt += len(targets)
                    batch_count += 1
                    
                    # Explicitly delete tensors to free memory
                    del inputs, targets, predictions, batch_loss
                    if 'descent_loss' in locals():
                        del descent_loss
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

            # Calculate epoch metrics
            if cnt > 0 and batch_count > 0:
                loss /= cnt
                accuracy /= batch_count
                print(f"Epoch: {epoch}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}")
                epoch_loss_train.append(loss)
                epoch_acc_train.append(accuracy)
            else:
                print(f"Epoch: {epoch}, No valid batches processed")
                epoch_loss_train.append(0)
                epoch_acc_train.append(0)

            # Validation phase
            model.eval()
            val_loss = 0.
            val_accuracy = 0.
            val_cnt = 0.
            val_batch_count = 0
            
            with torch.no_grad():
                for val_batch_idx, (inputs, targets) in enumerate(tqdm(val_generator, desc="Validation")):
                    try:
                        # Move data to device
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        
                        # Forward pass with mixed precision if available
                        if use_amp:
                            with autocast():
                                predictions = model(inputs.float())
                                batch_loss = criterion(predictions, targets).sum().item()
                        else:
                            predictions = model(inputs.float())
                            batch_loss = criterion(predictions, targets).sum().item()
                        
                        # Calculate accuracy
                        batch_accuracy = compute_accuracy(predictions.detach().cpu(), targets.cpu(), inf_threshold)
                        
                        # Update counters
                        val_loss += batch_loss
                        val_accuracy += batch_accuracy
                        val_cnt += len(targets)
                        val_batch_count += 1
                        
                        # Free memory
                        del inputs, targets, predictions
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"Error in validation batch {val_batch_idx}: {str(e)}")
                        continue
            
            # Calculate validation metrics
            if val_cnt > 0 and val_batch_count > 0:
                val_loss /= val_cnt
                val_accuracy /= val_batch_count
                print(f"Epoch: {epoch}, Val accuracy: {val_accuracy:6.2f} %, Val loss: {val_loss:8.5f}")
                epoch_loss_val.append(val_loss)
                epoch_acc_val.append(val_accuracy)
                
                # Update learning rate based on scheduler type
                if use_plateau_scheduler:
                    scheduler.step(val_accuracy)
                else:
                    # Use cosine annealing scheduler
                    scheduler.step()
                    
                    # Check if we should switch to plateau scheduler
                    if epoch > max_epochs // 2 and len(epoch_acc_val) > 3:
                        # Check if validation accuracy is plateauing
                        recent_accs = epoch_acc_val[-3:]
                        if max(recent_accs) - min(recent_accs) < 0.5:  # Less than 0.5% change
                            print("Validation accuracy plateauing. Switching to ReduceLROnPlateau scheduler.")
                            use_plateau_scheduler = True
                            plateau_scheduler.step(val_accuracy)
                
                # Early stopping check
                current_val_metric = val_accuracy
                if current_val_metric > best_val_metric:
                    best_val_metric = current_val_metric
                    early_stopping_counter = 0
                    # Save best model based on validation accuracy
                    torch.save(model.state_dict(), PATH + exp + '_best_val_ckpt.pt')
                    print(f"Saved new best model with validation accuracy: {val_accuracy:6.2f} %")
                else:
                    early_stopping_counter += 1
                    print(f"Validation accuracy did not improve. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
                    if early_stopping_counter >= early_stopping_patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        break
            else:
                print(f"Epoch: {epoch}, No valid validation batches processed")
                epoch_loss_val.append(0)
                epoch_acc_val.append(0)

            # Save best model based on training accuracy
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), PATH + exp + '_best_train_ckpt.pt')
        except Exception as epoch_e:
            print(f"Error during epoch {epoch}: {str(epoch_e)}")
            traceback.print_exc()
            continue   # Skip the epoch and continue to the next one

    print(f"Best train accuracy: {best_accuracy}")
    if has_validation:
        print(f"Best validation accuracy: {best_val_accuracy}")
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
                    print("Very limited GPU memory available for testing.")
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
        successful_batches = 0
        
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
                test_params['batch_size'] = 1  # Conservative default
        
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
        
        # Set a timeout for the entire testing process
        start_time = time.time()
        max_test_time = 1800  # 30 minutes max
        
        # Perform testing with no gradient computation
        with torch.no_grad():
            for test_batch_idx, (inputs, targets) in enumerate(tqdm(test_generator, desc="Testing")):
                # Check if we've exceeded the maximum test time
                if time.time() - start_time > max_test_time:
                    print(f"Testing has been running for over {max_test_time/60:.1f} minutes. Stopping early.")
                    break
                
                try:
                    # Move data to device
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    # Clear cache before forward pass
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # Forward pass with timeout protection
                    try:
                        # Set a timeout for the forward pass
                        forward_start = time.time()
                        forward_timeout = 30  # 30 seconds max for forward pass
                        
                        # Start forward pass
                        predictions = model(inputs.float())
                        
                        # Check if forward pass took too long
                        if time.time() - forward_start > forward_timeout:
                            print(f"Forward pass took too long ({time.time() - forward_start:.1f}s). Skipping this batch.")
                            continue
                    except Exception as forward_e:
                        print(f"Error during forward pass: {forward_e}")
                        traceback.print_exc()
                        continue
                    
                    # Calculate metrics
                    try:
                        batch_loss = criterion(predictions, targets).sum().item()
                        batch_accuracy = compute_accuracy(predictions.detach().cpu(), targets.cpu(), inf_threshold)
                    except Exception as metric_e:
                        print(f"Error calculating metrics: {metric_e}")
                        traceback.print_exc()
                        # Use default values
                        batch_loss = 0.0
                        batch_accuracy = 0.0
                    
                    # Update counters
                    test_loss += batch_loss
                    test_accuracy += batch_accuracy
                    cnt += len(targets)
                    successful_batches += 1
                    
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
                            try:
                                print(torch.cuda.memory_summary(device=device))
                            except:
                                print("Could not print CUDA memory summary")
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                        # If this is not the first batch and we're using GPU, try to move to CPU
                        if test_batch_idx > 0 and device.type == 'cuda':
                            print("Moving model to CPU to continue testing...")
                            try:
                                model = model.cpu()
                                device = torch.device("cpu")
                                continue
                            except Exception as cpu_e:
                                print(f"Error moving model to CPU: {cpu_e}")
                        elif test_params['batch_size'] > 1:
                            # Try with smaller batch size
                            test_params['batch_size'] = 1
                            print(f"Reducing batch size to {test_params['batch_size']} and recreating test data loader")
                            try:
                                test_generator = DataLoader(train_dataset, **test_params)
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
                        with open(PATH + "test_results.txt", "w") as f:
                            f.write(f"Model: {exp}_ckpt.pt\n")
                            f.write(f"Accuracy: {test_accuracy:6.2f} %\n")
                            f.write(f"Loss: {test_loss:8.5f}\n")
                            f.write(f"Samples: {cnt}\n")
                            f.write(f"Batches: {successful_batches}\n")
                            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        print(f"Test results saved to {PATH}test_results.txt")
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
    except Exception as test_e:
        print("Error during testing:", str(test_e))
        traceback.print_exc()
        print("Testing failed, but training was completed successfully.")
    finally:
        # Clean up resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Testing phase completed.")

    try:
        # Save visualization
        get_plot(PATH, epoch_acc_train, epoch_acc_val, 'Accuracy-' + exp, 'Train Accuracy', 'Val Accuracy', 'Epochs', 'Acc')
        get_plot(PATH, epoch_loss_train, epoch_loss_val, 'Loss-' + exp, 'Train Loss', 'Val Loss', 'Epochs', 'Loss')
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
