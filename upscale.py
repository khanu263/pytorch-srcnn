# upscale.py
# by Umair Khan
# CS 410 - Spring 2020

# Use a trained SRCNN to perform upscaling (or don't).

# Imports
import sys
import os
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Define the transfrom from Image to tensor
TO_TENSOR = transforms.ToTensor()

# Build argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-z", "--zoom", type = int, required = True)
parser.add_argument("-e", "--epochs", type = int, required = True)
parser.add_argument("-i", "--input", required = True)
parser.add_argument("-o", "--output", required = True)
parser.add_argument("-s", "--source")
parser.add_argument("-b", "--bicubic", default = False, action = "store_true")

# Parse and check arguments
args = parser.parse_args()
model_path = "models/{}_z{}.pt".format(args.epochs, args.zoom)
if not os.path.isfile(model_path):
    sys.exit("This model doesn't exist yet -- use train.py first.")
elif not os.path.isfile(args.input):
    sys.exit("The input path is not valid.")
elif args.source and not os.path.isfile(args.source):
    sys.exit("The source path is not valid.")

# Load the model
model = torch.load(model_path, map_location = torch.device('cpu'))
print("Loaded model.")

# Open input, do initial upscale, and extract channels
initial = Image.open(args.input).convert("YCbCr")
initial = initial.resize((int(args.zoom * initial.size[0]), int(args.zoom * initial.size[1])), Image.BICUBIC)
initial_y, initial_cb, initial_cr = initial.split()
print("Processed input image.")

# Determine if the request is for pure bicubic or SRCNN
if args.bicubic:

    # The outputs have already been made
    output = initial.convert("RGB")
    output_y = TO_TENSOR(initial_y)
    output_y = output_y.unsqueeze(0)
    print("Bicubic upsampling specified -- no work to do!")

else:

    # Convert Y channel to tensor and pass through network
    input_y = TO_TENSOR(initial_y).view(1, -1, initial_y.size[1], initial_y.size[0])
    output_y = model(input_y)
    print("Fed image through network.")

    # Convert back to image
    output = output_y[0].detach().numpy() * 255.0
    output = output.clip(0, 255)
    output = Image.fromarray(np.uint8(output[0]), mode = "L")

    # Merge back with the Cb and Cr channels
    output = Image.merge("YCbCr", [output, initial_cb, initial_cr]).convert("RGB")
    print("Merged channels.")

# If a source has been specified, we need to compute PSNR
if args.source:

    # Get the Y channel of the source
    source = Image.open(args.source).convert("YCbCr")
    source_y, source_cb, source_cr = source.split()
    source_y = TO_TENSOR(source_y)
    source_y = source_y.unsqueeze(0)
    print("Loaded source image.")

    # Create PyTorch criterion and calculate
    criterion = nn.MSELoss()
    loss = criterion(output_y, source_y)
    psnr = 10 * math.log10(1 / loss.item())
    print("PSNR: {:.2f}".format(psnr))

    # Create suffix for filename
    suffix = "_{:.2f}".format(psnr).replace(".", "-")

else:

    # Otherwise, there's nothing to do and no suffix
    suffix = ""

# Save output
split = args.output.split(".")
outfile = "{}{}.{}".format(split[0], suffix, split[1])
output.save(outfile)
print("Saved to {}".format(outfile))