from matplotlib.pyplot import get
import torch
import numpy as np
from Model.VideoSWIN import VideoSWIN3D
from my_dataloader import TinyVIRAT_dataset, SingleVideoDataset
from Preprocessing import get_prtn
from asam import ASAM
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.visualize import get_plot
from sklearn.metrics import accuracy_score
import os
import multiprocessing

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
    # Check for MPS (Metal) or CPU
    if torch.backends.mps.is_available():
        print("Using MPS (Metal)....")
        device = torch.device("mps")
    else:
        print("Using CPU....")
        device = torch.device("cpu")

    # Training Parameters
    shuffle = True
    print("Creating params....")
    params = {'batch_size': 2,
              'shuffle': shuffle,
              'num_workers': 0}  # Set to 0 to avoid multiprocessing issues

    max_epochs = 50
    inf_threshold = 0.6
    print(params)

    ############ Data Generators ############
    # Comment out or remove the original calls to get_prtn:
    # train_list_IDs, train_labels, train_IDs_path = get_prtn('train')
    # train_dataset = TinyVIRAT_dataset(list_IDs=train_list_IDs, labels=train_labels, IDs_path=train_IDs_path)
    # training_generator = DataLoader(train_dataset, **params)

    # Set the path to your single video file
    single_video_path = '/Users/mrterrific/Documents/Tinyactions/TinyActions/video.mp4'
    # Create the dataset directly
    train_dataset = SingleVideoDataset(video_path=single_video_path)
    training_generator = DataLoader(train_dataset, **params)

    # Optionally, you may want to also override the validation set or remove it entirely,
    # depending on your experiment setup.

    # Initialize the model
    model = VideoSWIN3D(
        num_classes=26,
        patch_size=(2,4,4),
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8,7,7),
        mlp_ratio=4.
    ).to(device)

    # Load pre-trained weights if available
    pretrained_path = '/path/to/your/pretrained/weights.pth'
    if os.path.exists(pretrained_path):
        model.load_pretrained_weights(pretrained_path)
    else:
        print("No pre-trained weights found. Training from scratch.")

    # Define loss and optimizer
    lr = 0.02
    wt_decay = 5e-4
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wt_decay)

    # ASAM
    rho = 0.55
    eta = 0.01
    minimizer = ASAM(optimizer, model, rho=rho, eta=eta)

    # Training and validation loops
    epoch_loss_train = []
    epoch_loss_val = []
    epoch_acc_train = []
    epoch_acc_val = []

    best_accuracy = 0.
    print("Begin Training....")
    for epoch in range(max_epochs):
        # Train
        model.train()
        loss = 0.
        accuracy = 0.
        cnt = 0.

        for batch_idx, (inputs, targets) in enumerate(tqdm(training_generator)):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Ascent Step
            predictions = model(inputs.float())
            batch_loss = criterion(predictions, targets)
            batch_loss.mean().backward()
            minimizer.ascent_step()

            # Descent Step
            criterion(model(inputs.float()), targets).mean().backward()
            minimizer.descent_step()

            with torch.no_grad():
                loss += batch_loss.sum().item()
                accuracy += compute_accuracy(predictions, targets, inf_threshold)
            cnt += len(targets)  # number of samples

        loss /= cnt
        accuracy /= (batch_idx + 1)
        print(f"Epoch: {epoch}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}")
        epoch_loss_train.append(loss)
        epoch_acc_train.append(accuracy)

        # Validation
        model.eval()
        loss = 0.
        accuracy = 0.
        cnt = 0.
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_generator):
                inputs = inputs.to(device)
                targets = targets.to(device)
                predictions = model(inputs.float())
                loss += criterion(predictions, targets).sum().item()
                accuracy += compute_accuracy(predictions, targets, inf_threshold)
                cnt += len(targets)

            loss /= cnt
            accuracy /= (batch_idx + 1)

        if best_accuracy < accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), PATH + exp + '_best_ckpt.pt')

        print(f"Epoch: {epoch}, Test accuracy: {accuracy:6.2f} %, Test loss: {loss:8.5f}")
        epoch_loss_val.append(loss)
        epoch_acc_val.append(accuracy)

    print(f"Best test accuracy: {best_accuracy}")
    print("TRAINING COMPLETED :)")

    # Save visualization
    get_plot(PATH, epoch_acc_train, epoch_acc_val, 'Accuracy-' + exp, 'Train Accuracy', 'Val Accuracy', 'Epochs', 'Acc')
    get_plot(PATH, epoch_loss_train, epoch_loss_val, 'Loss-' + exp, 'Train Loss', 'Val Loss', 'Epochs', 'Loss')

    # Save trained model
    torch.save(model, exp + "_ckpt.pt")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
