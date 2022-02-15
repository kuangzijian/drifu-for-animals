from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import torch
from PIL.ImageFilter import GaussianBlur
import glob


class TrainDataset_Stage2(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def default_loader(path):
        return Image.open(path).convert('RGB')

    def __init__(self, opt, phase='train', loader=default_loader):
        self.opt = opt
        self.loader = loader

        # Path setup
        self.root = self.opt.stage2dataroot
        if phase== 'train':
            self.im_list = glob.glob(os.path.join(self.root, 'train', '*/*.jpg'))
            self.class_dir = glob.glob(os.path.join(self.root, 'train', '*'))
        else:
            self.im_list = glob.glob(os.path.join(self.root, 'test', '*/*.jpg'))
            self.class_dir = glob.glob(os.path.join(self.root, 'test', '*'))

        self.imgs = [(im_path, self.class_dir.index(os.path.dirname(im_path))) for
                     im_path in self.im_list]
        random.shuffle(self.imgs)

        self.is_train = (phase == 'train')
        self.load_size = self.opt.loadSize

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])

    def __len__(self):
        return len(self.imgs)

    def get_item(self, index):
        '''
                Return the render data
                :param subject: subject name
                :param num_views: how many views to return
                :param view_id: the first view_id. If None, select a random one.
                :return:
                    'img': [num_views, C, W, H] images
                    'mask': [num_views, 1, W, H] masks
                '''
        render_list = []
        mask_list = []
        img_path, label = self.imgs[index]
        render = self.loader(img_path)
        seg_path = img_path.replace('.jpg', '.png')
        mask = Image.open(seg_path).convert('L')
        if self.is_train:
            # Pad images
            pad_size = int(0.1 * self.load_size)
            render = ImageOps.expand(render, pad_size, fill=0)
            mask = ImageOps.expand(mask, pad_size, fill=0)

            w, h = render.size
            th, tw = self.load_size, self.load_size

            # random flip
            if self.opt.random_flip and np.random.rand() > 0.5:
                render = transforms.RandomHorizontalFlip(p=1.0)(render)
                mask = transforms.RandomHorizontalFlip(p=1.0)(mask)

            # random scale
            if self.opt.random_scale:
                rand_scale = random.uniform(0.9, 1.1)
                w = int(rand_scale * w)
                h = int(rand_scale * h)
                render = render.resize((w, h), Image.BILINEAR)
                mask = mask.resize((w, h), Image.NEAREST)

            # random translate in the pixel space
            if self.opt.random_trans:
                dx = random.randint(-int(round((w - tw) / 10.)),
                                    int(round((w - tw) / 10.)))
                dy = random.randint(-int(round((h - th) / 10.)),
                                    int(round((h - th) / 10.)))
            else:
                dx = 0
                dy = 0

            x1 = int(round((w - tw) / 2.)) + dx
            y1 = int(round((h - th) / 2.)) + dy

            render = render.crop((x1, y1, x1 + tw, y1 + th))
            mask = mask.crop((x1, y1, x1 + tw, y1 + th))

            render = self.aug_trans(render)

            # random blur
            if self.opt.aug_blur > 0.00001:
                blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                render = render.filter(blur)
        else:
            # Pad images
            pad_size = int(0.1 * self.load_size)
            render = ImageOps.expand(render, pad_size, fill=0)
            mask = ImageOps.expand(mask, pad_size, fill=0)

            w, h = render.size
            th, tw = self.load_size, self.load_size
            x1 = int(round((w - tw) / 2.))
            y1 = int(round((h - th) / 2.))

            render = render.crop((x1, y1, x1 + tw, y1 + th))
            mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        mask = transforms.Resize(self.load_size)(mask)
        mask = transforms.ToTensor()(mask).float()
        mask_list.append(mask)

        render = self.to_tensor(render)
        render = mask.expand_as(render) * render

        render_list.append(render)

        return {
            'img': torch.stack(render_list, dim=0),
            'mask': torch.stack(mask_list, dim=0),
            'name': img_path.split('/')[-1].replace('.jpg', '')
        }

    def __getitem__(self, index):
        return self.get_item(index)