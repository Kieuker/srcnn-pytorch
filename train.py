import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn

from tqdm import tqdm
from datetime import datetime
import logging
import os
import skimage.color as sc
import numpy as np
import argparse

from datasets import T91Dataset
from datasets import Set5Dataset
from models import SRCNN
from utils.misc import RepeatedDataset
from utils.misc import crop_tensor_border
from utils.metrics import psnr
from utils.visualizations import display_set5_sr

device = "cuda" if torch.cuda.is_available() else "cpu"

# Default configuration values
experiment_path = './experiments' # Directory path to export experiment results.
scale_factor = 3
epochs = 2500
learning_rate = 0.00001
model_path = None
training_data_path = '/path/to/your/training/dataset/'
test_data_path = '/path/to/your/validation/dataset/'

# Argument Parser configuration
parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", type=str, help="Set directory path for storing experiment result files.")
parser.add_argument("--scale_factor", type=int, help="Set scale factor for super resolution.")
parser.add_argument("--learning_rate", type=float, help="Set learning rate for optimizer.")
parser.add_argument("--model_path", type=str, help="File path of .pt/.pth that stores parameters of model.")
parser.add_argument("--epochs", type=int, help="Max number of epochs for training.")
parser.add_argument("--training_data", type=str, help="Path for training data.")
parser.add_argument("--validation_data", type=str, help="Path for validation data.")

args = parser.parse_args()

if args.experiment_dir: experiment_path = args.experiment_dir
if args.scale_factor: scale_factor = args.scale_factor
if args.learning_rate: learning_rate = args.learning_rate
if args.model_path: model_path = args.model_path
if args.epochs: epochs = args.epochs
if args.training_data: training_data_path = args.training_data
if args.validation_data: test_data_path = args.validation_data

print("Experiment Directory Path: {}".format(experiment_path))
print("Scale Factor: {}".format(scale_factor))
print("Number of Epochs: {}".format(epochs))
print("Learning Rate: {}".format(learning_rate))

# Get current time for being used as experiment ID.
training_period = datetime.now().strftime("%y%m%d-%H%M%S")

# Logger configuration
logging.basicConfig(filename=os.path.join(experiment_path, f'{training_period}.scale-{scale_factor}.log'),
                    filemode='w',
                    format='%(asctime)s %(levelname)s:%(message)s',
                    level=logging.INFO,
                    )

logging.info("Experiment Directory Path: {}".format(experiment_path))
logging.info("Scale Factor: {}".format(scale_factor))
logging.info("Number of Epochs: {}".format(epochs))
logging.info("Learning Rate: {}".format(learning_rate))
# dataset
training_data = T91Dataset(training_data_path, scale_factor=scale_factor, color_mode='RGB', crop_gt_border_by_n=(9//2)+(5//2))
test_data = Set5Dataset(test_data_path, scale_factor=scale_factor, color_mode='RGB', crop_gt_border_by_n=(9//2)+(5//2))

# Inflate iteration per epoch via RepeatedDataset
training_data = RepeatedDataset(training_data, 3516)

# DataLoader
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=1)

for X, y in test_dataloader:
    X, y = X.to(device), y.to(device)
    print("Shape of X:{}".format(X.shape))
    print("Shape of y:{}".format(y.shape))
    break

# Model configuration
model = SRCNN(do_padding=False, in_channels=3).to(device)
if model_path:
    model.load_state_dict(torch.load(model_path, weights_only=True))
    logging.info("Model loaded from {}".format(model_path))
print(model)

# Define loss function and optimizer.
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training
def train(dataloader, model, loss_fn, optimizer):
    running_loss = 0.0
    model.train()
    pbar = tqdm(dataloader)
    for batch, (X, y) in enumerate(pbar):
        X, y = X.to(device), y.to(device)

        # Calcutale loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item()
        running_loss += loss
        pbar.set_postfix(loss = running_loss/(pbar.n+1))
    running_loss /= len(dataloader)

    return running_loss
# Evaluation
def test(test_dataloader, model):
    model.eval()
    final_psnr = 0
    bicubic_psnr = 0
    image_list = []
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            pred = torch.clamp(pred, min=0., max=1.)

            X = torch.squeeze(X, dim=0)
            y = torch.squeeze(y, dim=0)
            pred = torch.squeeze(pred, dim=0)

            X = crop_tensor_border(X, (9//2)+(5//2))

            # Store PIL images
            lr_image = transforms.ToPILImage()(X)
            sr_image = transforms.ToPILImage()(pred)
            gt_image = transforms.ToPILImage()(y)
            image_list.append([lr_image, sr_image, gt_image])

            # Preperation for Grayscale PSNR
            X = transforms.ToPILImage()(X)
            y = transforms.ToPILImage()(y)
            pred = transforms.ToPILImage()(pred)
            X = sc.rgb2ycbcr(np.float32(X) / 255)[:, :, 0]
            y = sc.rgb2ycbcr(np.float32(y) / 255)[:, :, 0]
            pred = sc.rgb2ycbcr(np.float32(pred) / 255)[:, :, 0]
            X = transforms.ToTensor()(X) / 255
            y = transforms.ToTensor()(y) / 255
            pred = transforms.ToTensor()(pred) / 255

            # Calculate PSNR
            final_psnr += psnr(pred, y)
            bicubic_psnr += psnr(X, y)

        final_psnr /= len(test_dataloader)
        bicubic_psnr /= len(test_dataloader)
    print("Validation PSNR: {}".format(final_psnr))
    print("Bicubic PSNR   : {}".format(bicubic_psnr))

    return image_list, final_psnr, bicubic_psnr

# Initializtion for monitoring best loss and psnr scores.
min_loss = 100.
best_psnr = -1.

# Main loop - training
for epoch in range(epochs):
    print("Epoch: {:3d}".format(epoch + 1))
    loss = train(train_dataloader, model, loss_fn, optimizer)
    image_list, final_psnr, bicubic_psnr = test(test_dataloader, model)
    logging.info(f"Epoch: {epoch+1} - Loss: {loss:.10f}, PSNR: {final_psnr:.10f}, bicubic_PSNR: {bicubic_psnr:.10f}")

    # Save model parameters of epochs that scores Best loss and best PSNR.
    if min_loss > loss:
        torch.save(model.state_dict(), os.path.join(experiment_path, f"{training_period}.scale-{scale_factor}.MIN_LOSS.pth"))
        min_loss = loss
    if best_psnr < final_psnr:
        torch.save(model.state_dict(), os.path.join(experiment_path, f"{training_period}.scale-{scale_factor}.BEST_PSNR.pth"))
        logging.info(f"Best PSNR is renewed from [{best_psnr:.10f}] to [{final_psnr:.10f}]! (Epoch {epoch+1})")
        best_psnr = final_psnr

    # Save model parameters every 1000th epochs
    if (epoch+1) % 1000 == 0:
        display_set5_sr(image_list, export_dir=experiment_path, export_filename=f'{training_period}.scale-{scale_factor}.epoch-{epoch+1}.png', is_grayscale=False)  
        logging.info(f"Image exported to '{training_period}.scale-{scale_factor}.epoch-{epoch+1}.png'.")
        torch.save(model.state_dict(), os.path.join(experiment_path, f"{training_period}.scale-{scale_factor}.epoch-{epoch+1}.pth"))

torch.save(model.state_dict(), os.path.join(experiment_path, f"{training_period}.scale-{scale_factor}.LATEST.epoch-{epoch+1}.pth"))

with torch.no_grad():
    model.load_state_dict(torch.load(os.path.join(experiment_path, f"{training_period}.scale-{scale_factor}.BEST_PSNR.pth"), weights_only=True))
    image_list, final_psnr, bicubic_psnr = test(test_dataloader, model)
    print(f'Best validation PSNR (scale:{scale_factor}): {final_psnr}')
    logging.info(f'Best validation PSNR (scale:{scale_factor}): {final_psnr}')
    display_set5_sr(image_list, export_dir=experiment_path, export_filename=f'{training_period}.scale-{scale_factor}.BEST_PSNR.png', is_grayscale=False)

print("Done!")
logging.info("Training Done!")