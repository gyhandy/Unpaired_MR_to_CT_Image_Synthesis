import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image

from torchvision.transforms import functional as F

class SingleDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        # self.dir_A = os.path.join(opt.dataroot)
        self.dir_A = os.path.join(opt.dataroot,
                                  opt.phase + 'A')  # depend on the dataroot name make dataset /MR2CT/testA
        self.A_paths = make_dataset(self.dir_A)

        # self.A_paths = sorted(self.A_paths)
        self.A_paths = sorted(self.A_paths,
                              key=lambda x: int(x.split('_')[-1].split('.')[-2]))  # sort as increasing of i :1, 2, 3...

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)
        # '''
        # Crop the given PIL Image.
        # crop(img, i, j, h, w)
        #
        # Args:
        #     img (PIL Image): Image to be cropped.
        #     i: Upper pixel coordinate.
        #     j: Left pixel coordinate.
        #     h: Height of the cropped image.
        #     w: Width of the cropped image.
        # '''
        # if self.opt.test_crop_mode == '0':
        #     A = F.crop(A, 0, 0, 256, 256)
        # elif self.opt.test_crop_mode == '1':
        #     A = F.crop(A, 0, 30, 256, 256)
        # elif self.opt.test_crop_mode == '2':
        #     A = F.crop(A, 30, 0, 256, 256)
        # elif self.opt.test_crop_mode == '3':
        #     A = F.crop(A, 30, 30, 256, 256)


        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
        else:
            input_nc = self.opt.input_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'
