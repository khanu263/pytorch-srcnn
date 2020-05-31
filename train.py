# train.py
# by Umair Khan
# CS 410 - Spring 2020

# Train the SRCNN at the given zoom level, keeping
# track of loss over time.

# Imports
import sys
import os
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import SRDataset
from model import SRCNN

# Import matplotlib and seaborn (X workaround) and style
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_context("talk")
matplotlib.rcParams["font.family"] = "Public Sans"
matplotlib.rcParams["font.weight"] = "semibold"
matplotlib.rcParams["axes.titleweight"] = "bold"
matplotlib.rcParams["axes.labelweight"] = "medium"
matplotlib.rcParams["axes.titlesize"] = "medium"
matplotlib.rcParams["axes.labelsize"] = "small"
matplotlib.rcParams["axes.labelpad"] = 20.0
matplotlib.rcParams['xtick.labelsize'] = "small"
matplotlib.rcParams['ytick.labelsize'] = "small"
matplotlib.rcParams['text.color'] = "#333333"
matplotlib.rcParams['axes.labelcolor'] = "#333333"
matplotlib.rcParams['xtick.color'] = "#333333"
matplotlib.rcParams['ytick.color'] = "#333333"
matplotlib.rcParams["axes.facecolor"] = "white"
matplotlib.rcParams["axes.edgecolor"] = "#333333"
matplotlib.rcParams["lines.linewidth"] = 2.5
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False

# Build argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-z", "--zoom", type = int, required = True)
parser.add_argument("-e", "--epochs", type = int, required = True)
parser.add_argument("-b", "--batch-size", type = int, required = True)
parser.add_argument("-c", "--cuda", default = False, action = "store_true")
parser.add_argument("-r", "--resume")

# Parse and check arguments
args = parser.parse_args()
if args.zoom < 1 or args.epochs < 1 or args.batch_size < 1:
    sys.exit("Zoom factor, epoch count, and batch size must be at least 1.")

# Select computation device
device = torch.device("cuda:0" if (args.cuda and torch.cuda.is_available()) else "cpu")

# Load model to continue training
if args.resume:

    # Check for file validity
    if not os.path.isfile("models/" + args.resume + ".pt"):
        sys.exit("Specified model does not exist.")

    # Load model
    model = torch.load("models/" + args.resume + ".pt")
    print("Loaded SRCNN model.")

# Create new model if not specified
else:
    model = SRCNN()
    print("Created new SRCNN model.")

# Shift model to computation device
model = model.to(device)
print("Moved model to device {}.".format(device))

# Load datasets
train_data = SRDataset("data/train/", args.zoom)
val_data = SRDataset("data/val/", args.zoom)
test_data = SRDataset("data/test/", args.zoom)
print("Loaded datasets.")

# Make data loaders
train_loader = DataLoader(dataset = train_data, batch_size = args.batch_size, shuffle = True)
val_loader = DataLoader(dataset = val_data, batch_size = args.batch_size, shuffle = False)
test_loader = DataLoader(dataset = test_data, batch_size = args.batch_size, shuffle = False)
print("Created data loaders with batch size {}.".format(args.batch_size))

# Define the loss function and per-layer optimization, as per the paper
criterion = nn.MSELoss()
optimizer = optim.SGD([
                        {"params": model.patch_ex.parameters(), "lr": 0.0001},
                        {"params": model.nl_mapping.parameters(), "lr": 0.0001},
                        {"params": model.reconstruction.parameters(), "lr": 0.00001}
                      ], lr = 0.0001, momentum = 0.9)
print("Defined loss and optimization.")

# Load metric lists from last training run
if args.resume:

    # Check for file validity
    if not os.path.isfile("psnr/" + args.resume + ".txt"):
        sys.exit("Specified metric file does not exist.")

    # Read lines from file
    lines = open("psnr/" + args.resume + ".txt", "r").read().splitlines()
    avg_train_psnrs = [float(x) for x in lines[0].split("\t")[1:]]
    avg_val_psnrs = [float(x) for x in lines[1].split("\t")[1:]]
    offset = int(args.resume.split("_")[0])
    print("Loaded metric lists.")

# Otherwise, initialize new metric lists
else:
    avg_train_psnrs = []
    avg_val_psnrs = []
    offset = 0
    print("Initialized new metric lists.")

# Go through each epoch
for e in range(args.epochs):

    # Start timing
    t0 = time.time()

    # Initialize PSNR accumulators
    train_psnr = 0
    val_psnr = 0

    # Go through each training example
    for batch in train_loader:

        # Extract example and zero out optimizer
        x, t = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()

        # Feed example and backpropagate
        y = model(x)
        loss = criterion(y, t)
        loss.backward()
        optimizer.step()

        # Calculate and accumulate PSNR
        train_psnr += 10 * math.log10(1 / loss.item())

    # Go through each validation example (turn off gradient tracking)
    with torch.no_grad():
        for batch in val_loader:

            # Extract instance and feed through model
            x, t = batch[0].to(device), batch[1].to(device)
            y = model(x)
            loss = criterion(y, t)

            # Calculate and accumulate PSNR
            val_psnr += 10 * math.log10(1 / loss.item())

    # Store results
    avg_train_psnrs.append(train_psnr / len(train_loader))
    avg_val_psnrs.append(val_psnr / len(val_loader))

    # End timing
    t1 = time.time()

    # Print results every five epochs
    if e % 5 == 0:
        print("Finished epoch {}. Average training PSNR: {:.2f}. Average validation PSNR: {:.2f}. Time for epoch: {:.2f} sec."
              .format(e + offset, avg_train_psnrs[-1], avg_val_psnrs[-1], t1 - t0))

# Print final training metrics
print("Finished training. Final training PSNR: {:.2f}. Final validation PSNR: {:.2f}."
      .format(avg_train_psnrs[-1], avg_val_psnrs[-1]))

# Initialize testing metric
test_psnr = 0

# Go through each training example and accumulate metrics
with torch.no_grad():
    for batch in test_loader:
        x, t = batch[0].to(device), batch[1].to(device)
        y = model(x)
        loss = criterion(y, t)
        test_psnr += 10 * math.log10(1 / loss.item())

# Print testing results
print("Finished testing. Average PSNR: {:.2f}."
      .format(test_psnr / len(test_loader)))

# Get filename for current model being trained
base_file = "{}_z{}".format(args.epochs + offset, args.zoom)

# Plot training and validation PSNR
x_ax = list(range(1, args.epochs + offset + 1))
plt.figure(figsize = (10, 6))
plt.xlabel("Epoch")
plt.ylabel("Average PSNR (dB)")
plt.plot(x_ax, avg_val_psnrs, "tab:blue", label = "validation")
plt.plot(x_ax, avg_train_psnrs, "tab:orange", label = "training")
plt.legend(loc = "best")
plt.tight_layout()
plt.savefig("figs/{}.png".format(base_file))
print("Saved PSNR plot.")

# Save raw PSNR data
output_lines = ["train\t" + "\t".join(map(str, avg_train_psnrs)),
                "valid\t" + "\t".join(map(str, avg_val_psnrs)),
                "test\t" + str(test_psnr / len(test_loader))]
with open("psnr/{}.txt".format(base_file), "w") as f:
    f.write("\n".join(output_lines))
print("Saved raw PSNR data.")

# Save SRCNN model
torch.save(model, "models/{}.pt".format(base_file))
print("Saved PyTorch model.")