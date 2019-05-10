import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class UnalignedMaskDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.dir_A_mask = os.path.join(opt.dataroot, opt.phase + 'A_mask')
        self.dir_B_mask = os.path.join(opt.dataroot, opt.phase + 'B_mask')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_mask_paths = make_dataset(self.dir_A_mask)
        self.B_mask_paths = make_dataset(self.dir_B_mask)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)

        self.A_mask_paths = sorted(self.A_mask_paths)
        self.B_mask_paths = sorted(self.B_mask_paths)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        index_A = index % self.A_size
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)

        A_path = self.A_paths[index_A]
        B_path = self.B_paths[index_B]
        A_mask_path = self.A_mask_paths[index_A]
        B_mask_path = self.B_mask_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_mask_img = Image.open(A_mask_path).convert('L')
        B_mask_img = Image.open(B_mask_path).convert('L')

        r, g, b = A_img.split()
        A_img_cat = Image.merge("RGBA", [r,g,b,A_mask_img])

        r, g, b = B_img.split()
        B_img_cat = Image.merge("RGBA", [r, g, b, B_mask_img])

        A_cat = self.transform(A_img_cat)
        B_cat = self.transform(B_img_cat)



        A = A_cat[0:3, :, :]
        B = B_cat[0:3, :, :]

        A_mask = A_cat[3, :, :]
        B_mask = B_cat[3, :, :]

        A_mask = A_mask.unsqueeze(0)
        B_mask = B_mask.unsqueeze(0)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path,
                'A_mask':A_mask, 'B_mask':B_mask}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedMaskDataset'
