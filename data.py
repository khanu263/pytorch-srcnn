# data.py
# by Umair Khan
# CS 410 - Spring 2020

# Define a custom PyTorch dataset incorporating the
# channel manipulation and transformations described
# in the original paper.

# Imports
import glob
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image, ImageFilter

# Sub-image crop size, as defined in paper
CROP = 33

# Class definition
class SRDataset(Dataset):

    # Dataset setup.
    # Arguments:
    #  - dirname - directory to generate dataset from
    #  - zoom - zoom factor for the dataset
    def __init__(self, dirname, zoom):

        # Initialize superclass
        super(SRDataset, self).__init__()

        # Derive a valid crop from the zoom factor
        crop = CROP - (CROP % zoom)

        # Get a list of filepaths for images in folder
        self.files = glob.glob(dirname + "*")

        # Define transforms for network inputs
        # (subsample and interpolate to synthesize low-resolution)
        self.tf_in = transforms.Compose([transforms.CenterCrop(crop),
                                         transforms.Resize(crop // zoom),
                                         transforms.Resize(crop, interpolation = Image.BICUBIC),
                                         transforms.ToTensor()])

        # Define transforms from network outputs
        # (no manipulation needed)
        self.tf_out = transforms.Compose([transforms.CenterCrop(crop),
                                          transforms.ToTensor()])

    # Function to load the Y channel (YCbCr) of the image from the given filepath.
    # Arguments:
    #  - filepath - path to image file
    def load(self, filepath):
        raw = Image.open(filepath).convert("YCbCr")
        y, cb, cr = raw.split()
        return y

    # Hook function to retrieve item from dataset.
    # Arguments:
    #  - i - index to retrieve
    def __getitem__(self, i):

        # Load the image
        input_img = self.load(self.files[i])
        output_img = input_img.copy()

        # Apply transformations
        # (paper specifies Gaussian blur on input)
        input_img = input_img.filter(ImageFilter.GaussianBlur(1))
        input_img = self.tf_in(input_img)
        output_img = self.tf_out(output_img)

        # Return results
        return input_img, output_img

    # Hook function to retreive dataset length.
    def __len__(self):
        return len(self.files)
