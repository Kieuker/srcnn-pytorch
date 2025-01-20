import torch
import argparse
import os
from tqdm import tqdm
from PIL import Image
# from torchvision.transforms.v2 import ToPILImage, PILToTensor
from torchvision.transforms import ToPILImage, ToTensor

from models import SRCNN

# Add arguments
parser = argparse.ArgumentParser()
parser.add_argument('--f') # for vscode interactive window.
parser.add_argument('-m', "--model", type=str, default='./pretrained_models/scale-3.BEST_PSNR.pretrained.pth',
                    help="Path to your trained model.")
parser.add_argument('-s', "--scale", type=int, default=3, help='The scaling factor you want to apply.')
parser.add_argument('-i', "--images", type=str, default='./inference/input/', help='Directory path to your images.')
parser.add_argument('-o', "--output", type=str, default='./inference/results/',
                    help='The directory path for the upscaled images.')

args = parser.parse_args()
model_path = args.model
scale = args.scale
input_images_directory = args.images
output_results_directory = args.output

# Device config
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = SRCNN(do_padding=False, in_channels=3).to(device)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# Open images from the directory
valid_extensions = tuple(Image.registered_extensions().keys())
image_filenames = [filename for filename in os.listdir(input_images_directory)
                   if filename.lower().endswith(valid_extensions)]
image_paths = [os.path.join(input_images_directory, filename) for filename in image_filenames]
images = [Image.open(path) for path in image_paths]

# Upsample images
for idx in tqdm(range(len(images))):
    image = images[idx]
    w, h = image.size
    image = image.resize((w*scale, h*scale), Image.BICUBIC)
    image = ToTensor()(image)
    image = image.to(device)
    with torch.no_grad():
        result = model(image)
        result = torch.clamp(result, min=0., max=1.)
        ToPILImage()(result).save(os.path.join(output_results_directory, image_filenames[idx]))
    