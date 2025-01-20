import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from concurrent.futures import ThreadPoolExecutor
import numpy as np

class Set5Dataset(Dataset):
    def __init__(self, image_dir, scale_factor = 3, color_mode = 'RGB', crop_gt_border_by_n=None):
        self.image_dir = image_dir
        self.scale_factor = scale_factor
        self.image_names = os.listdir(image_dir)
        self.color_mode = color_mode
        self.crop_gt_border = crop_gt_border_by_n
        self.image_paths = self._get_image_paths()
        self.images = self._preload_images()

        # If color_mode value is 'YCbCr', use Y-channel value from the image only
        self._is_grayscale = False
        if self.color_mode == 'YCbCr': self._is_grayscale = True

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        GT_image = transforms.ToPILImage()(image)

        width, height = GT_image.size

        LR_image = GT_image.resize((width // self.scale_factor, height // self.scale_factor))
        LR_image = LR_image.resize((width, height), Image.BICUBIC)

        lr = transforms.ToTensor()(LR_image)
        gt = transforms.ToTensor()(GT_image)

        # Process image base on the options: _is_grayscale, crop_gt_border
        if self._is_grayscale: lr, gt = lr[0].unsqueeze(0), gt[0].unsqueeze(0)
        if self.crop_gt_border: gt = self._crop_tensor_border(gt, self.crop_gt_border)
        
        return lr, gt

    def _preload_images(self):
        print(f"Preloading...({self.__class__.__name__})")
        with ThreadPoolExecutor() as executor:
            images = list(executor.map(self._load_image_as_numpy, self.image_paths))
        print(f"Preloading Done!({self.__class__.__name__})")
        return images
    
    def _load_image_as_numpy(self, img_path):
            return np.array(Image.open(img_path).convert(self.color_mode))
    
    def _get_image_paths(self):
        images_paths = list()
        for image_name in self.image_names:
            images_paths.append(os.path.join(self.image_dir, image_name))
        return images_paths

    
    def _crop_tensor_border(self, tensor, n):
        _, h, w = tensor.shape
        
        cropped_tensor = tensor[:, n:h-n, n:w-n]
        
        return cropped_tensor