import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np


class OCTDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # get the image directory
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(
            self.dir_AB, opt.max_dataset_size))  # get image paths
        # crop_size should be smaller than the size of loaded image
        assert (self.opt.load_size >= self.opt.crop_size)
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = transforms.Compose([
            transforms.ToTensor(),
            Conver2Uint8(),
            MyResize((256, 256)),
            ToOneHot(self.input_nc)
        ])
        B_transform = get_transform(
            self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = np.asarray(A_transform(A), dtype=np.float32)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)


class Conver2Uint8(torch.nn.Module):
    '''
    Resize input when the target dim is not divisible by the input dim
    '''

    def __init__(self):
        super().__init__()

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        img = torch.round(torch.mul(img, 255))
        return img


class MyResize(torch.nn.Module):
    '''
    Resize input when the target dim is not divisible by the input dim
    '''

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        h, w = img.shape[-2], img.shape[-1]
        target_h, target_w = self.size
        assert h % target_h == 0, f"target_h({target_h}) must be divisible by h({h})"
        assert w % target_w == 0, f"target_w({target_w}) must be divisible by w({w})"
        # Resize by assigning the max value of each pixel grid
        kernel_h = h // target_h
        kernel_w = w // target_w
        img_target = torch.nn.functional.max_pool2d(
            img, kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
        return img_target


class ToOneHot(torch.nn.Module):
    '''
    Convert input to one-hot encoding
    '''

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, img):
        """
        Args:
                img (Tensor): Image to be scaled of shape (1, h, w).

        Returns:
                Tensor: Rescaled image.
        """
        img = img.long()[0]
        img = torch.nn.functional.one_hot(img, num_classes=self.num_classes)
        img = img.permute(2, 0, 1)
        return img
