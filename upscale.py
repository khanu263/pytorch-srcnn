# upscale.py
# by Umair Khan
# CS 410 - Spring 2020

# Use a trained SRCNN to perform upscaling.

# Imports
import sys
import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image

# Build argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-z", "--zoom", type = int, required = True)
parser.add_argument("-i", "--input", required = True)
parser.add_argument("-o", "--output", required = True)

# Parse and check arguments
args = parser.parse_args()
model_path = "models/zoom_{}.pt".format(args.zoom)
if not os.path.isfile(model_path):
    sys.exit("This model hasn't been trained yet -- use train.py first.")
elif not os.path.isfile(args.input):
    sys.exit("The input path is not valid.")

# Load the model
model = torch.load(model_path)
print("Loaded model.")

# Open input, do initial upscale, and extract channels
initial = Image.open(args.input).convert("YCbCr")
initial = x.resize((int(args.zoom * x.size[0]), int(args.zoom * x.size[1])), Image.BICUBIC)
y, cb, cr = img.split()

# Convert input image (Y channel) to tensor
initial_tensor = transforms.ToTensor(y).view(1, -1, y.size[1], y.size[0])
print("Processed input image.")

# Pass through network
output_tensor = model(initial_tensor)
print("Fed image through network.")

# Convert back to image
output = output_tensor[0].detach().numpy() * 255.0
output = output.clip(0, 255)
output = Image.fromarray(np.uint8(output[0]), mode = "L")

# Merge back with the Cb and Cr channels
output = Image.merge("YCbCr", [output, cb, cr]).convert("RGB")
print("Merged channels.")

# Save output
output.save(args.output)
print("Saved to output location.")