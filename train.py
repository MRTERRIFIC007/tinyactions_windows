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

    # Device selection: Use CUDA if available, check for MPS, else default to CPU
    try:
        if torch.cuda.is_available():
            try:
                # Test if CUDA is actually working by creating a small tensor
                test_tensor = torch.zeros(1).cuda()
                del test_tensor  # Free memory
                print("Using CUDA in train.py....")
                device = torch.device("cuda")
            except RuntimeError as e:
                print(f"CUDA is available but encountered an error: {e}")
                print("Falling back to CPU...")
                device = torch.device("cpu")
        elif torch.backends.mps.is_available():
            print("Using MPS (Metal) in train.py....")
            device = torch.device("mps")
        else:
            print("Using CPU in train.py....")
            device = torch.device("cpu")
    except Exception as e:
        print("Error in device selection in train.py, defaulting to CPU:", e)
        device = torch.device("cpu")

    # If using CUDA, enable pin_memory in DataLoader for faster data transfer
    if device.type == 'cuda':
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
        # Initialize the model with standard parameters
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
        
        # Try to move model to the selected device, fall back to CPU if there's an error
        try:
            model = model.to(device)
        except RuntimeError as cuda_err:
            if device.type == 'cuda':
                print(f"Error moving model to CUDA: {cuda_err}")
                print("Falling back to CPU...")
                device = torch.device("cpu")
                model = model.to(device)
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
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wt_decay)

        # ASAM
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
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()

                    # Ascent Step
                    predictions = model(inputs.float())
                    batch_loss = criterion(predictions, targets)
                    batch_loss.mean().backward()
                    minimizer.ascent_step()

                    # Descent Step
                    descent_loss = criterion(model(inputs.float()), targets)
                    descent_loss.mean().backward()
                    minimizer.descent_step()

                    with torch.no_grad():
                        loss += batch_loss.sum().item()
                        accuracy += compute_accuracy(predictions, targets, inf_threshold)
                    cnt += len(targets)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f"CUDA out of memory encountered on batch {batch_idx} at epoch {epoch}. Clearing cache, collecting garbage, and printing memory summary.")
                        if torch.cuda.is_available():
                            print(torch.cuda.memory_summary(device=device))
                        torch.cuda.empty_cache()
                        gc.collect()
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

    # Testing phase on the same single video dataset
    try:
        print("Testing on the entire training dataset...")
        model.eval()
        test_loss = 0.
        test_accuracy = 0.
        cnt = 0.
        with torch.no_grad():
            for test_batch_idx, (inputs, targets) in enumerate(tqdm(training_generator, desc="Testing")):
                try:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    predictions = model(inputs.float())
                    test_loss += criterion(predictions, targets).sum().item()
                    test_accuracy += compute_accuracy(predictions, targets, inf_threshold)
                    cnt += len(targets)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f"CUDA out of memory encountered on test batch {test_batch_idx}. Clearing cache, collecting garbage, and printing memory summary.")
                        if torch.cuda.is_available():
                            print(torch.cuda.memory_summary(device=device))
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        raise e
                except Exception as e:
                    print(f"Error in test batch {test_batch_idx}: {str(e)}")
                    traceback.print_exc()
                    continue
            test_loss /= cnt
            test_accuracy /= (test_batch_idx + 1)
        print(f"Test metrics - Accuracy: {test_accuracy:6.2f} %, Loss: {test_loss:8.5f}")
    except Exception as test_e:
        print("Error during testing:", str(test_e))
        traceback.print_exc()

    try:
        # Save visualization
        get_plot(PATH, epoch_acc_train, None, 'Accuracy-' + exp, 'Train Accuracy', 'Val Accuracy (N/A)', 'Epochs', 'Acc')
        get_plot(PATH, epoch_loss_train, None, 'Loss-' + exp, 'Train Loss', 'Val Loss (N/A)', 'Epochs', 'Loss')
    except Exception as e:
        print("Error while plotting:", str(e))
        traceback.print_exc()

    try:
        # Save trained model
        torch.save(model, exp + "_ckpt.pt")
    except Exception as e:
        print("Error while saving the trained model:", str(e))
        traceback.print_exc()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
