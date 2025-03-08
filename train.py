from matplotlib.pyplot import get
import torch
import numpy as np
from Model.VideoSWIN import VideoSWIN3D
from my_dataloader import TinyVIRAT_dataset, get_video_data
from asam import ASAM
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.visualize import get_plot
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import multiprocessing
import traceback
import gc
import time
import random
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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

# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Set seeds for reproducibility
set_seed(42)

exp = '23'

# Make exp dir
if not os.path.exists('exps/exp_' + exp + '/'):
    os.makedirs('exps/exp_' + exp + '/')
PATH = 'exps/exp_' + exp + '/'

def compute_accuracy(pred, target, inf_th):
    """
    Compute accuracy with error handling and additional metrics.
    Returns accuracy, precision, recall, and F1 score.
    """
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
            return 0.0, 0.0, 0.0, 0.0
        
        # Calculate accuracy
        acc = accuracy_score(target, pred_binary)
        
        # Calculate precision, recall, and F1 score (macro average)
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                target, pred_binary, average='macro', zero_division=0
            )
        except Exception as e:
            print(f"Error calculating precision/recall: {e}")
            precision, recall, f1 = 0.0, 0.0, 0.0
            
        return acc, precision, recall, f1
    except Exception as e:
        print(f"Error in compute_accuracy: {e}")
        traceback.print_exc()
        return 0.0, 0.0, 0.0, 0.0  # Return default values on error

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
                device = torch.device("cpu")
                # Update epochs for CPU training
                max_epochs = 100
                early_stopping_patience = 15
                print(f"Updated to CPU training with {max_epochs} epochs and {early_stopping_patience} early stopping patience")
                return device
        
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

def plot_training_progress(epochs, train_metrics, val_metrics, metric_name, save_path):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        epochs: List of epoch numbers
        train_metrics: List of training metrics
        val_metrics: List of validation metrics
        metric_name: Name of the metric (e.g., 'Loss', 'Accuracy')
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metrics, 'b-', label=f'Training {metric_name}')
    if val_metrics:
        plt.plot(epochs, val_metrics, 'r-', label=f'Validation {metric_name}')
    
    plt.title(f'Training and Validation {metric_name} Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    
    # Force x-axis to be integers for epochs
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {metric_name} plot to {save_path}")

def update_training_plots(epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, path):
    """
    Update and save training progress plots after each epoch.
    
    Args:
        epoch: Current epoch number
        train_loss, train_acc, train_f1: Lists of training metrics
        val_loss, val_acc, val_f1: Lists of validation metrics
        path: Directory to save plots
    """
    epochs = list(range(1, epoch + 2))  # +2 because epoch is 0-indexed and we want to include current epoch
    
    # Plot loss
    plot_training_progress(
        epochs, train_loss, val_loss, 'Loss', 
        os.path.join(path, 'loss_progress.png')
    )
    
    # Plot accuracy
    plot_training_progress(
        epochs, train_acc, val_acc, 'Accuracy (%)', 
        os.path.join(path, 'accuracy_progress.png')
    )
    
    # Plot F1 score
    plot_training_progress(
        epochs, train_f1, val_f1, 'F1 Score', 
        os.path.join(path, 'f1_progress.png')
    )
    
    # Create a combined metrics plot
    plt.figure(figsize=(12, 10))
    
    # Loss subplot
    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    if val_loss:
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy subplot
    plt.subplot(3, 1, 2)
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    if val_acc:
        plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # F1 subplot
    plt.subplot(3, 1, 3)
    plt.plot(epochs, train_f1, 'b-', label='Training F1')
    if val_f1:
        plt.plot(epochs, val_f1, 'r-', label='Validation F1')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'combined_metrics.png'))
    plt.close()
    print(f"Saved combined metrics plot to {os.path.join(path, 'combined_metrics.png')}")

def main():
    # Set up error logging to file
    error_log_path = os.path.join(PATH, "error_log.txt")
    try:
        error_log = open(error_log_path, "w")
        # Redirect stderr to the error log file
        sys.stderr = error_log
        print(f"Redirecting error logs to {error_log_path}")
    except Exception as e:
        print(f"Could not set up error logging: {e}")
        error_log = None
    
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
        
        # Set threshold for binary classification
        inf_threshold = 0.5  # Reduced from 0.6 for better recall
        
        # Get the optimal device first, before setting epochs
        device = get_optimal_device()
        
        # Set epochs based on the actual device that will be used
        if device.type == 'cpu':
            max_epochs = 100  # More epochs for CPU training
            early_stopping_patience = 15  # Increased patience for CPU training
            print(f"Using CPU training with {max_epochs} epochs and {early_stopping_patience} early stopping patience")
        else:
            # GPU training
            try:
                gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
                if gpu_memory < 4:
                    max_epochs = 15  # For GPUs with less than 4GB memory
                    early_stopping_patience = 7
                    print(f"Using GPU training with limited memory ({gpu_memory:.2f} GB). Setting {max_epochs} epochs.")
                else:
                    max_epochs = 20  # For GPUs with more memory
                    early_stopping_patience = 7
                    print(f"Using GPU training with sufficient memory ({gpu_memory:.2f} GB). Setting {max_epochs} epochs.")
            except Exception as e:
                print(f"Error determining GPU memory: {e}")
                max_epochs = 15  # Conservative default
                early_stopping_patience = 7
                print(f"Using default GPU training with {max_epochs} epochs.")
        
        early_stopping_counter = 0
        best_val_metric = 0
        
        params = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': 0,  # Set to 0 to avoid multiprocessing issues
            'pin_memory': False,  # Disable pin_memory by default to avoid errors
            'drop_last': False  # Process all samples
        }
        
        print(f"Training parameters: batch_size={batch_size}, grad_accumulation={grad_accumulation_steps}, epochs={max_epochs}, early_stopping_patience={early_stopping_patience}, inf_threshold={inf_threshold}")
    except Exception as e:
        print("Error during training parameter initialization:", str(e))
        traceback.print_exc()
        if error_log:
            error_log.close()
            sys.stderr = sys.__stderr__
        return

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
    else:
        # Ensure pin_memory is disabled for CPU
        params['pin_memory'] = False
        print("pin_memory disabled for CPU training")

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
            # Use stratified split if possible to maintain class distribution
            try:
                from sklearn.model_selection import train_test_split
                
                # Convert labels to format suitable for stratified split
                label_indices = {}
                for i, id in enumerate(list_IDs):
                    # Use the first non-zero label as the class
                    label = labels[id]
                    label_idx = next((i for i, x in enumerate(label) if x > 0), 0)
                    label_indices[id] = label_idx
                
                # Get unique labels
                unique_labels = set(label_indices.values())
                
                if len(unique_labels) > 1:
                    # Use stratified split
                    print("Using stratified split for validation set")
                    train_list_IDs, val_list_IDs = train_test_split(
                        list_IDs, 
                        test_size=0.2, 
                        random_state=42,
                        stratify=[label_indices[id] for id in list_IDs]
                    )
                else:
                    # Fall back to random split
                    print("Using random split for validation set (only one class detected)")
                    val_size = max(1, int(len(list_IDs) * 0.2))
                    train_size = len(list_IDs) - val_size
                    train_list_IDs = list_IDs[:train_size]
                    val_list_IDs = list_IDs[train_size:]
            except Exception as e:
                print(f"Error creating stratified split: {e}. Using random split.")
                val_size = max(1, int(len(list_IDs) * 0.2))
                train_size = len(list_IDs) - val_size
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
            # For CPU, use optimized model with efficient attention
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
                # Update epochs for CPU training
                max_epochs = 100
                early_stopping_patience = 15
                print(f"Updated to CPU training with {max_epochs} epochs and {early_stopping_patience} early stopping patience")
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
                if device.type == 'cuda':
                    print(f"Error moving smaller model to CUDA: {cuda_err}")
                    print("Falling back to CPU...")
                    device = torch.device("cpu")
                    # Update epochs for CPU training
                    max_epochs = 100
                    early_stopping_patience = 15
                    print(f"Updated to CPU training with {max_epochs} epochs and {early_stopping_patience} early stopping patience")
                    model = model.to(device)
                else:
                    raise  # Re-raise if it's not a CUDA device
            except Exception as e2:
                print(f"Error initializing smaller model: {e2}")
                print("Falling back to CPU with minimal model...")
                device = torch.device("cpu")
                # Update epochs for CPU training
                max_epochs = 100
                early_stopping_patience = 15
                print(f"Updated to CPU training with {max_epochs} epochs and {early_stopping_patience} early stopping patience")
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
        except Exception as e2:
            print(f"Error initializing smaller model: {e2}")
            print("Falling back to CPU with minimal model...")
            device = torch.device("cpu")
            # Update epochs for CPU training
            max_epochs = 100
            early_stopping_patience = 15
            print(f"Updated to CPU training with {max_epochs} epochs and {early_stopping_patience} early stopping patience")
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
            T_0 = max(5, max_epochs // 5)  # Restart every T_0 epochs, adjusted for CPU training
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
            if device.type == 'cuda':
                minimizer = ASAM(optimizer, model, rho=rho, eta=eta)
                use_asam = True
                print(f"Using ASAM optimizer with rho={rho}, eta={eta}")
            else:
                # Skip ASAM for CPU training to avoid CUDA tensor errors
                use_asam = False
                print("ASAM optimizer disabled for CPU training to avoid CUDA tensor errors")
        except Exception as e:
            print(f"Error initializing ASAM: {e}. Using standard optimizer.")
            use_asam = False
    except Exception as e:
        print("Error during loss, optimizer, or ASAM initialization:", str(e))
        traceback.print_exc()
        if error_log:
            error_log.close()
            sys.stderr = sys.__stderr__
        return

    # Training and validation loops
    epoch_loss_train = []
    epoch_acc_train = []
    epoch_f1_train = []
    epoch_loss_val = []
    epoch_acc_val = []
    epoch_f1_val = []

    best_accuracy = 0.
    best_val_accuracy = 0.
    best_val_f1 = 0.
    print("Begin Training....")
    
    # Save training configuration for reproducibility
    try:
        config_path = os.path.join(PATH, "training_config.txt")
        with open(config_path, "w") as config_file:
            config_file.write(f"Device: {device}\n")
            config_file.write(f"Batch size: {batch_size}\n")
            config_file.write(f"Gradient accumulation steps: {grad_accumulation_steps}\n")
            config_file.write(f"Max epochs: {max_epochs}\n")
            config_file.write(f"Early stopping patience: {early_stopping_patience}\n")
            config_file.write(f"Inference threshold: {inf_threshold}\n")
            config_file.write(f"Learning rate: {lr}\n")
            config_file.write(f"Weight decay: {wt_decay}\n")
            config_file.write(f"Model embed_dim: {model.embed_dim if hasattr(model, 'embed_dim') else 'N/A'}\n")
            config_file.write(f"Use ASAM: {use_asam}\n")
            config_file.write(f"Use AMP: {use_amp}\n")
            config_file.write(f"Training samples: {len(train_dataset) if 'train_dataset' in locals() else 'N/A'}\n")
            config_file.write(f"Validation samples: {len(val_dataset) if 'val_dataset' in locals() and has_validation else 'N/A'}\n")
            config_file.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"Saved training configuration to {config_path}")
    except Exception as e:
        print(f"Error saving training configuration: {e}")
    
    # Training loop with error handling
    for epoch in range(max_epochs):
        try:
            epoch_start_time = time.time()
            
            # Train
            model.train()
            loss = 0.
            accuracy = 0.
            precision = 0.
            recall = 0.
            f1 = 0.
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
                    if hasattr(model, 'set_grad_checkpointing') and device.type == 'cuda':
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
                            if use_asam and device.type == 'cuda':  # Only use ASAM with CUDA
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
                        # Calculate accuracy, precision, recall, and F1 score
                        acc, prec, rec, f1_score = compute_accuracy(predictions.detach().cpu(), targets.cpu(), inf_threshold)
                        accuracy += acc
                        precision += prec
                        recall += rec
                        f1 += f1_score
                    cnt += len(targets)
                    batch_count += 1
                    
                    # Explicitly delete tensors to free memory
                    del inputs, targets, predictions, batch_loss
                    if 'descent_loss' in locals():
                        del descent_loss
                    if 'descent_predictions' in locals():
                        del descent_predictions
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
                precision /= batch_count
                recall /= batch_count
                f1 /= batch_count
                print(f"Epoch: {epoch}, Train accuracy: {accuracy:6.2f} %, Precision: {precision:6.2f}, Recall: {recall:6.2f}, F1: {f1:6.2f}, Loss: {loss:8.5f}")
                epoch_loss_train.append(loss)
                epoch_acc_train.append(accuracy)
                epoch_f1_train.append(f1)
            else:
                print(f"Epoch: {epoch}, No valid batches processed")
                epoch_loss_train.append(0)
                epoch_acc_train.append(0)
                epoch_f1_train.append(0)

            # Validation phase
            model.eval()
            val_loss = 0.
            val_accuracy = 0.
            val_precision = 0.
            val_recall = 0.
            val_f1 = 0.
            val_cnt = 0.
            val_batch_count = 0
            
            # Skip validation if no validation data is available
            if has_validation:
                try:
                    with torch.no_grad():
                        for val_batch_idx, (inputs, targets) in enumerate(tqdm(val_generator)):
                            try:
                                # Move data to device
                                inputs = inputs.to(device)
                                targets = targets.to(device)
                                
                                # Free up memory before forward pass
                                if device.type == 'cuda':
                                    torch.cuda.empty_cache()
                                
                                # Forward pass
                                predictions = model(inputs.float())
                                batch_loss = criterion(predictions, targets).sum().item()
                                
                                # Calculate metrics
                                acc, prec, rec, f1_score = compute_accuracy(predictions.detach().cpu(), targets.cpu(), inf_threshold)
                                
                                # Update counters
                                val_loss += batch_loss
                                val_accuracy += acc
                                val_precision += prec
                                val_recall += rec
                                val_f1 += f1_score
                                val_cnt += len(targets)
                                val_batch_count += 1
                                
                                # Explicitly delete tensors to free memory
                                del inputs, targets, predictions
                                if device.type == 'cuda':
                                    torch.cuda.empty_cache()
                                
                            except Exception as val_batch_e:
                                print(f"Error processing validation batch {val_batch_idx}: {str(val_batch_e)}")
                                traceback.print_exc()
                                # Skip this batch but continue validation
                                continue
                    
                    # Calculate validation metrics
                    if val_cnt > 0 and val_batch_count > 0:
                        val_loss /= val_cnt
                        val_accuracy /= val_batch_count
                        val_precision /= val_batch_count
                        val_recall /= val_batch_count
                        val_f1 /= val_batch_count
                        print(f"Epoch: {epoch}, Val accuracy: {val_accuracy:6.2f} %, Val precision: {val_precision:6.2f}, Val recall: {val_recall:6.2f}, Val F1: {val_f1:6.2f}, Val loss: {val_loss:8.5f}")
                        epoch_loss_val.append(val_loss)
                        epoch_acc_val.append(val_accuracy)
                        epoch_f1_val.append(val_f1)
                        
                        # Update learning rate based on scheduler type
                        if use_plateau_scheduler:
                            plateau_scheduler.step(val_accuracy)
                        else:
                            scheduler.step()
                            
                        # Early stopping check
                        if val_accuracy > best_val_accuracy:
                            best_val_accuracy = val_accuracy
                            best_val_f1 = val_f1
                            early_stopping_counter = 0
                            # Save best validation model
                            try:
                                torch.save(model.state_dict(), PATH + exp + '_best_val_ckpt.pt')
                                print(f"Saved best validation model with accuracy: {best_val_accuracy:6.2f}% and F1: {best_val_f1:6.2f}")
                            except Exception as save_e:
                                print(f"Error saving best validation model: {save_e}")
                                # Try saving to CPU if GPU save fails
                                try:
                                    cpu_model = model.cpu()
                                    torch.save(cpu_model.state_dict(), PATH + exp + '_best_val_cpu_ckpt.pt')
                                    print(f"Saved best validation model to CPU with accuracy: {best_val_accuracy:6.2f}%")
                                    model = model.to(device)  # Move back to original device
                                except Exception as cpu_save_e:
                                    print(f"Error saving validation model to CPU: {cpu_save_e}")
                        else:
                            early_stopping_counter += 1
                            print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
                            
                        # Check for early stopping
                        if early_stopping_counter >= early_stopping_patience:
                            print(f"Early stopping triggered after {epoch+1} epochs")
                            # Break out of the epoch loop
                            break
                            
                        # Switch scheduler strategy if validation performance plateaus
                        if epoch > 10 and len(epoch_acc_val) > 5:
                            # Check if validation accuracy has plateaued
                            recent_vals = epoch_acc_val[-5:]
                            if max(recent_vals) - min(recent_vals) < 0.5:  # Less than 0.5% change
                                if not use_plateau_scheduler:
                                    print("Validation accuracy has plateaued. Switching to ReduceLROnPlateau scheduler.")
                                    use_plateau_scheduler = True
                    else:
                        print(f"Epoch: {epoch}, No valid validation batches processed")
                        epoch_loss_val.append(0)
                        epoch_acc_val.append(0)
                        epoch_f1_val.append(0)
                        
                except Exception as val_e:
                    print(f"Error during validation phase: {str(val_e)}")
                    traceback.print_exc()
                    # Continue with next epoch even if validation fails
                    epoch_loss_val.append(epoch_loss_val[-1] if len(epoch_loss_val) > 0 else 0)
                    epoch_acc_val.append(epoch_acc_val[-1] if len(epoch_acc_val) > 0 else 0)
                    epoch_f1_val.append(epoch_f1_val[-1] if len(epoch_f1_val) > 0 else 0)
            else:
                # No validation data available
                print(f"Epoch: {epoch}, No validation data available")
                epoch_loss_val.append(0)
                epoch_acc_val.append(0)
                epoch_f1_val.append(0)
                
                # Update learning rate without validation metrics
                scheduler.step()
                
            # Calculate epoch duration
            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds")
            
            # Update and save training plots after each epoch
            update_training_plots(
                epoch, 
                epoch_loss_train, epoch_acc_train, epoch_f1_train,
                epoch_loss_val, epoch_acc_val, epoch_f1_val,
                PATH
            )
            
            # Periodically save checkpoint to prevent complete loss in case of crash
            if epoch % 5 == 0 or epoch == max_epochs - 1:
                try:
                    checkpoint_path = PATH + exp + f'_epoch_{epoch}_ckpt.pt'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                        'best_accuracy': best_accuracy,
                        'best_val_accuracy': best_val_accuracy,
                        'best_val_f1': best_val_f1,
                        'early_stop_counter': early_stopping_counter,
                    }, checkpoint_path)
                    print(f"Saved checkpoint at epoch {epoch} to {checkpoint_path}")
                except Exception as ckpt_e:
                    print(f"Error saving checkpoint at epoch {epoch}: {ckpt_e}")

            # Save best model based on training accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), PATH + exp + '_best_train_ckpt.pt')
        except Exception as epoch_e:
            print(f"Error during epoch {epoch}: {str(epoch_e)}")
            traceback.print_exc()
            # Skip the epoch and continue to the next one
            continue

    print(f"Best train accuracy: {best_accuracy}")
    if has_validation:
        print(f"Best validation accuracy: {best_val_accuracy}")
        print(f"Best validation F1 score: {best_val_f1}")

    # Save final model
    try:
        final_model_path = PATH + exp + '_final_ckpt.pt'
        torch.save(model.state_dict(), final_model_path)
        print(f"Saved final model to {final_model_path}")
    except Exception as e:
        print(f"Error saving final model: {e}")
        try:
            # Try saving to CPU if GPU save fails
            cpu_model = model.cpu()
            cpu_model_path = PATH + exp + '_final_cpu_ckpt.pt'
            torch.save(cpu_model.state_dict(), cpu_model_path)
            print(f"Saved final model to CPU at {cpu_model_path}")
            model = model.to(device)  # Move back to original device
        except Exception as cpu_e:
            print(f"Error saving final model to CPU: {cpu_e}")
            print("WARNING: Final model was not saved!")

    # Test the model on test data if available
    if test_dataset is not None and len(test_dataset) > 0:
        print("\nEvaluating model on test data...")
        try:
            # Load the best validation model if available
            best_model_path = PATH + exp + '_best_val_ckpt.pt'
            if os.path.exists(best_model_path):
                print(f"Loading best validation model from {best_model_path}")
                model.load_state_dict(torch.load(best_model_path, map_location=device))
            else:
                print("Best validation model not found. Using final model for testing.")
            
            # Set model to evaluation mode
            model.eval()
            
            # Initialize test metrics
            test_loss = 0.
            test_accuracy = 0.
            test_precision = 0.
            test_recall = 0.
            test_f1 = 0.
            test_cnt = 0
            test_batch_count = 0
            
            # Store predictions and targets for confusion matrix
            all_predictions = []
            all_targets = []
            
            # Test loop with error handling
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(tqdm(test_generator, desc="Testing")):
                    try:
                        # Move data to device
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        
                        # Forward pass
                        predictions = model(inputs.float())
                        
                        # Calculate loss
                        batch_loss = criterion(predictions, targets).sum().item()
                        
                        # Calculate metrics
                        acc, prec, rec, f1_score = compute_accuracy(predictions.detach().cpu(), targets.cpu(), inf_threshold)
                        
                        # Store predictions and targets for later analysis
                        pred_binary = (torch.sigmoid(predictions.detach().cpu()) > inf_threshold).float()
                        all_predictions.append(pred_binary)
                        all_targets.append(targets.cpu())
                        
                        # Update counters
                        test_loss += batch_loss
                        test_accuracy += acc
                        test_precision += prec
                        test_recall += rec
                        test_f1 += f1_score
                        test_cnt += len(targets)
                        test_batch_count += 1
                        
                        # Free memory
                        del inputs, targets, predictions
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        print(f"Error processing test batch {batch_idx}: {e}")
                        traceback.print_exc()
                        continue
            
            # Calculate final test metrics
            if test_cnt > 0 and test_batch_count > 0:
                test_loss /= test_cnt
                test_accuracy /= test_batch_count
                test_precision /= test_batch_count
                test_recall /= test_batch_count
                test_f1 /= test_batch_count
                
                # Print test metrics
                print("\nTest Results:")
                print(f"Accuracy: {test_accuracy:6.2f}%")
                print(f"Precision: {test_precision:6.2f}")
                print(f"Recall: {test_recall:6.2f}")
                print(f"F1 Score: {test_f1:6.2f}")
                print(f"Loss: {test_loss:8.5f}")
                print(f"Samples: {test_cnt}")
                
                # Save test results to file
                try:
                    results_path = os.path.join(PATH, f"{exp}_test_results.txt")
                    with open(results_path, "w") as f:
                        f.write("Test Results:\n")
                        f.write(f"Accuracy: {test_accuracy:6.2f}%\n")
                        f.write(f"Precision: {test_precision:6.2f}\n")
                        f.write(f"Recall: {test_recall:6.2f}\n")
                        f.write(f"F1 Score: {test_f1:6.2f}\n")
                        f.write(f"Loss: {test_loss:8.5f}\n")
                        f.write(f"Samples: {test_cnt}\n")
                        f.write(f"Batches: {test_batch_count}\n")
                        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    print(f"Test results saved to {results_path}")
                    
                    # Try to generate per-class metrics
                    try:
                        # Concatenate all predictions and targets
                        all_pred_tensor = torch.cat(all_predictions, dim=0).numpy()
                        all_target_tensor = torch.cat(all_targets, dim=0).numpy()
                        
                        # Calculate per-class metrics
                        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
                            precision_recall_fscore_support(all_target_tensor, all_pred_tensor, average=None, zero_division=0)
                        
                        # Save per-class metrics
                        per_class_path = os.path.join(PATH, f"{exp}_per_class_metrics.txt")
                        with open(per_class_path, "w") as f:
                            f.write("Per-Class Metrics:\n")
                            f.write("Class\tPrecision\tRecall\tF1\tSupport\n")
                            for i in range(len(precision_per_class)):
                                f.write(f"{i}\t{precision_per_class[i]:.4f}\t{recall_per_class[i]:.4f}\t{f1_per_class[i]:.4f}\t{support_per_class[i]}\n")
                        print(f"Per-class metrics saved to {per_class_path}")
                    except Exception as per_class_e:
                        print(f"Error calculating per-class metrics: {per_class_e}")
                except Exception as results_e:
                    print(f"Error saving test results: {results_e}")
            else:
                print("No valid test batches processed")
        except Exception as test_e:
            print(f"Error during testing: {test_e}")
            traceback.print_exc()
    else:
        print("No test dataset available for evaluation")

    # Generate and save training plots
    try:
        print("Generating final training plots...")
        # Create traditional line plots
        get_plot(PATH, epoch_acc_train, epoch_acc_val, 'Accuracy-' + exp, 'Train Accuracy', 'Val Accuracy', 'Epochs', 'Acc')
        get_plot(PATH, epoch_loss_train, epoch_loss_val, 'Loss-' + exp, 'Train Loss', 'Val Loss', 'Epochs', 'Loss')
        get_plot(PATH, epoch_f1_train, epoch_f1_val, 'F1-' + exp, 'Train F1', 'Val F1', 'Epochs', 'F1')
        
        # Create a final combined plot with all metrics
        epochs = list(range(1, len(epoch_acc_train) + 1))
        
        plt.figure(figsize=(15, 12))
        
        # Loss subplot with dual y-axis
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(epochs, epoch_loss_train, 'b-', linewidth=2, label='Train Loss')
        if epoch_loss_val:
            ax1.plot(epochs, epoch_loss_val, 'r-', linewidth=2, label='Val Loss')
        ax1.set_title('Training and Validation Metrics', fontsize=16)
        ax1.set_ylabel('Loss', fontsize=14)
        ax1.legend(loc='upper right', fontsize=12)
        ax1.grid(True)
        
        # Accuracy subplot
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(epochs, epoch_acc_train, 'b-', linewidth=2, label='Train Accuracy')
        if epoch_acc_val:
            ax2.plot(epochs, epoch_acc_val, 'r-', linewidth=2, label='Val Accuracy')
        ax2.set_ylabel('Accuracy (%)', fontsize=14)
        ax2.legend(loc='lower right', fontsize=12)
        ax2.grid(True)
        
        # F1 score subplot
        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(epochs, epoch_f1_train, 'b-', linewidth=2, label='Train F1')
        if epoch_f1_val:
            ax3.plot(epochs, epoch_f1_val, 'r-', linewidth=2, label='Val F1')
        ax3.set_xlabel('Epochs', fontsize=14)
        ax3.set_ylabel('F1 Score', fontsize=14)
        ax3.legend(loc='lower right', fontsize=12)
        ax3.grid(True)
        
        # Force x-axis to be integers for epochs
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        final_plot_path = os.path.join(PATH, f'final_training_metrics_{exp}.png')
        plt.savefig(final_plot_path, dpi=300)
        plt.close()
        
        print(f"Successfully saved visualization plots. Final combined plot saved to {final_plot_path}")
    except Exception as plot_e:
        print(f"Error generating plots: {plot_e}")
        traceback.print_exc()

    # Close error log if it was opened
    if error_log:
        error_log.close()
        sys.stderr = sys.__stderr__
        print(f"Error log saved to {error_log_path}")

    # Print training summary
    print("\nTraining Summary:")
    print(f"Total epochs completed: {epoch + 1}")
    print(f"Best training accuracy: {best_accuracy:.2f}%")
    if has_validation:
        print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
        print(f"Best validation F1 score: {best_val_f1:.2f}")
    print(f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {(time.time() - start_time) / 60:.2f} minutes")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
